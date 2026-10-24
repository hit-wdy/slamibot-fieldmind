"""Phase A verification for the RADIO image/text encoder.

Loads the real pinned checkpoint and checks the things the spec says must be
verified rather than assumed: the vector contract on both sides, the adaptor
pairing, the preprocessing, the code/checkpoint revision, image-text ranking,
and the cost on this machine (load time, peak VRAM, per-image and per-query
latency). It writes a markdown report and exits non-zero if a hard check fails.

The image-text ranking is a wiring check, not an accuracy measurement: it says
the two sides share one space, not that retrieval works on real scenes. That is
phase E.

Usage:
    python scripts/verify_phase_a.py
    python scripts/verify_phase_a.py --resolutions 768x768 512 --runs 12
    python scripts/verify_phase_a.py --dtypes float32 bfloat16
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from slamibot_fieldmind import AlignedEncoder, Config, prepare_image
from slamibot_fieldmind.paths import PROJECT_ROOT

DEFAULT_RESOLUTIONS = ["768x768", "512", "768"]
REPORT_PATH = PROJECT_ROOT / "reports" / "phase_a_encoder.md"


def parse_resolution(text: str) -> list[int]:
    parts = [p for p in text.replace(",", "x").split("x") if p]
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad resolution {text!r}; use 768x768 or 512") from None
    if len(values) not in (1, 2) or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"bad resolution {text!r}; use 768x768 or 512")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resolutions", nargs="+", type=parse_resolution, default=None)
    parser.add_argument("--dtypes", nargs="+", choices=("auto", "float32", "bfloat16", "float16"), default=None)
    parser.add_argument("--runs", type=int, default=8, help="timed encodes per resolution")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--frame-size", type=int, nargs=2, default=[576, 1024], metavar=("H", "W"))
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    return parser.parse_args(argv)


def synthetic_frame(seed: int, size: tuple[int, int]) -> np.ndarray:
    """A deterministic stand-in for a camera frame, so results are repeatable."""
    height, width = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (127 + 100 * np.sin(x / 40.0 + seed)).astype(np.int16),
            (127 + 100 * np.cos(y / 30.0 - seed)).astype(np.int16),
            ((x * y / max(height * width, 1)) * 255).astype(np.int16),
        ],
        axis=-1,
    )
    noise = rng.integers(-12, 13, size=base.shape).astype(np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def gpu_memory_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "device_total_mb": round(total / 2**20, 1),
        "device_free_mb": round(free / 2**20, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def environment_section() -> list[str]:
    lines = [
        f"- date (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- torch: {torch.__version__} (CUDA {torch.version.cuda})",
        f"- CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        lines.append(f"- GPU: {torch.cuda.get_device_name(0)}")
        lines.append(f"- VRAM: {free / 2**20:.0f} MB free of {total / 2**20:.0f} MB")
        lines.append(f"- compute capability: {torch.cuda.get_device_capability(0)} (sm_86 = Ampere)")
        lines.append(f"- bf16 supported: {torch.cuda.is_bf16_supported()}")
        lines.append(f"- torch built for arch: {torch.cuda.get_arch_list()}")
    return lines


def load_section(encoder: AlignedEncoder, load_seconds: float) -> list[str]:
    report = encoder.load_report
    facts = encoder.facts
    lines = [
        f"- load time: {load_seconds:.2f} s",
        f"- RADIO version key: `{report.version}` (from the official RESOURCE_MAP)",
        f"- adaptor: `{report.adaptor}` -> `{report.adaptor_classes.get(report.adaptor)}`",
        f"- text encoder the adaptor is paired with: "
        f"`{report.text_encoder.get('model')}` / `{report.text_encoder.get('pretrained')}`"
        f" ({report.text_encoder.get('type')})",
        f"- checkpoint: `{report.checkpoint}`",
        f"- checkpoint sha256: `{report.checkpoint_sha256}` (verified, {report.checkpoint_bytes / 1e6:.1f} MB)",
        f"- RADIO source commit: `{report.source_commit}`",
        f"- checkpoint dtype {report.checkpoint_dtype} -> run dtype {report.model_dtype}",
        f"- parameters: {report.parameter_count / 1e6:.1f} M",
        f"- device {report.device} / text tower {report.text_encoder_device}",
        f"- patch size {facts['patch_size']}, min_resolution_step {facts['min_resolution_step']}, "
        f"max_resolution {facts['max_resolution']}, preferred {facts.get('preferred_resolution')}",
        f"- backbone embed_dim {facts['embed_dim']}, summary_dim {facts['summary_dim']}",
        f"- embedding dim from the adaptor head: {encoder.embedding_dim}",
        f"- fingerprint: `{encoder.fingerprint()}`",
        "",
        "Teachers declared by the checkpoint (`clip` is the one loaded here):",
        "",
        "| teacher | type | model | pretrained | contributes summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for teacher in report.checkpoint_teachers:
        lines.append(
            f"| {teacher['name']} | {teacher['type']} | {teacher['model']} | "
            f"{teacher['pretrained']} | {teacher['use_summary']} |"
        )
    lines.append("")
    if report.missing_keys or report.unexpected_keys:
        lines.append("State-dict reconciliation warnings (expected: training-only tensors are not loaded):")
        lines.append("")
        for warning in report.warnings:
            if "keys in state dict" in warning:
                lines.append(f"- `{warning}`")
    else:
        lines.append("No missing or unexpected state-dict keys.")
    lines.append("")
    return lines


def contract_section(encoder: AlignedEncoder, frame: np.ndarray) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    failures: list[str] = []

    single = encoder.encode_images(frame)
    double = encoder.encode_images([frame, frame])
    checks = [
        ("shape of one image is [1, D]", single.shape == (1, encoder.embedding_dim), str(single.shape)),
        ("dtype is float32", single.dtype == np.float32, str(single.dtype)),
        ("values are finite", bool(np.isfinite(single).all()), ""),
        (
            "L2 norm is 1",
            bool(np.allclose(np.linalg.norm(single, axis=1), 1.0, atol=1e-5)),
            f"{np.linalg.norm(single, axis=1).tolist()}",
        ),
        ("repeat encode is bit-identical", bool(np.array_equal(single, double[:1])), ""),
        ("same image cosine is 1", bool(float((double[0] @ double[1])) > 0.99999), ""),
    ]
    for name, ok, detail in checks:
        lines.append(f"- [{'x' if ok else ' '}] {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)
    return lines, failures


def measure(encoder: AlignedEncoder, frame: np.ndarray, runs: int, warmup: int) -> dict:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        encoder.encode_images(frame)

    timings = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        encoder.encode_images(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)

    prepared = prepare_image(frame, encoder.model_config.input_resolution, encoder.resize_multiple)
    height, width = prepared.shape[-2:]
    result = {
        "resolution": list(encoder.model_config.input_resolution),
        "model_input_size": [height, width],
        "tokens": (height // encoder.resize_multiple) * (width // encoder.resize_multiple),
        "ms_p50": round(statistics.median(timings), 2),
        "ms_p95": round(sorted(timings)[max(0, int(round(0.95 * len(timings))) - 1)], 2),
        "ms_min": round(min(timings), 2),
        "ms_max": round(max(timings), 2),
        "images_per_second": round(1000.0 / statistics.median(timings), 2),
        "running_dtype": encoder.load_report.model_dtype,
        "fingerprint_short": encoder.fingerprint_short(),
    }
    result.update(gpu_memory_mb())
    return result


def separation(encoder: AlignedEncoder, frames: list[np.ndarray]) -> dict:
    embeddings = encoder.encode_images(frames)
    similarity = embeddings @ embeddings.T
    off = similarity[~np.eye(len(frames), dtype=bool)]
    return {
        "min_off_diagonal": round(float(off.min()), 4),
        "max_off_diagonal": round(float(off.max()), 4),
        "mean_off_diagonal": round(float(off.mean()), 4),
    }


def distinct_frames(size: tuple[int, int]) -> list[np.ndarray]:
    """Frames sharing no structure, so the separation numbers mean something.

    A set of similar frames would collapse towards cosine 1 in any CLIP space and
    would say nothing about whether the encoder discriminates.
    """
    height, width = size
    rng = np.random.default_rng(1)
    red = np.zeros((height, width, 3), dtype=np.uint8)
    red[:, :, 0] = 220
    blue = np.zeros((height, width, 3), dtype=np.uint8)
    blue[:, :, 2] = 220
    white = np.full((height, width, 3), 245, dtype=np.uint8)
    checker = np.repeat((((np.indices((height, width)).sum(0) // 40) % 2) * 255)[:, :, None], 3, axis=2)
    y, x = np.mgrid[0:height, 0:width]
    gradient = np.stack([x * 255 // width, y * 255 // height, np.zeros_like(x)], axis=-1).astype(np.uint8)
    noise = rng.integers(0, 256, (height, width, 3)).astype(np.uint8)
    return [red, blue, white, checker.astype(np.uint8), gradient, noise]


def text_latency(encoder: AlignedEncoder, runs: int, warmup: int) -> dict:
    labels = [
        "a photo of a cat",
        "a dog",
        "a car",
        "a fire truck",
        "a bicycle",
        "a building",
        "a kitten",
        "a puppy",
        "a truck",
        "a bus",
    ]

    def median_ms(fn) -> float:
        for _ in range(warmup):
            fn()
        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - started) * 1000.0)
        return round(statistics.median(timings), 2)

    result = {
        "device": encoder.load_report.text_encoder_device,
        "batch_10_ms": median_ms(lambda: encoder.encode_texts(labels)),
        "single_ms": median_ms(lambda: encoder.encode_texts(["a fire truck"])),
    }
    result.update(gpu_memory_mb())
    return result


def text_alignment(encoder: AlignedEncoder, frame_size: tuple[int, int]) -> tuple[list[str], list[str]]:
    """Image-text ranking on frames whose content is known by construction.

    Falsifiable without any dataset: a solid red frame must score `a red image`
    above every other label. This is the spec 2.2 startup check, and it fails
    loudly if the image summary is not in the adaptor's text space.
    """
    height, width = frame_size
    red = np.zeros((height, width, 3), dtype=np.uint8)
    red[:, :, 0] = 220
    blue = np.zeros((height, width, 3), dtype=np.uint8)
    blue[:, :, 2] = 220
    checker = np.repeat((((np.indices((height, width)).sum(0) // 40) % 2) * 255)[:, :, None], 3, axis=2)
    frames = [red, blue, checker.astype(np.uint8)]
    labels = ["a red image", "a blue image", "a black and white checkered pattern"]

    images = encoder.encode_images(frames)
    texts = encoder.encode_texts(labels)
    scores = texts @ images.T

    lines = [
        "| frame | " + " | ".join(labels) + " | top label |",
        "| --- | " + " | ".join("---" for _ in labels) + " | --- |",
    ]
    failures = []
    for i, frame_name in enumerate(["solid red", "solid blue", "checkerboard"]):
        row = scores[:, i]
        lines.append(
            f"| {frame_name} | "
            + " | ".join(f"{float(value):.4f}" for value in row)
            + f" | {labels[int(np.argmax(row))]} |"
        )
        if int(np.argmax(row)) != i:
            failures.append(f"image-text ranking: {frame_name} matched {labels[int(np.argmax(row))]!r}")
    return lines, failures


def real_photo_alignment(encoder: AlignedEncoder) -> list[str]:
    """Informational: a real photo, when the repo has one to hand."""
    photo = PROJECT_ROOT / "tests" / "cat.png"
    if not photo.is_file():
        return ["No `tests/cat.png`; skipping the real-photo check.", ""]
    labels = ["a kitten", "a photo of a cat", "a puppy", "a dog", "a car", "a fire truck", "a building"]
    scores = encoder.encode_texts(labels) @ encoder.encode_images([photo])[0]
    lines = [
        f"Ranking against `{photo.relative_to(PROJECT_ROOT)}`:",
        "",
        "| label | cosine |",
        "| --- | --- |",
    ]
    order = np.argsort(-scores)
    for i in order:
        lines.append(f"| {labels[i]} | {float(scores[i]):.4f} |")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.from_yaml(args.config) if args.config else Config.default()
    dtypes = args.dtypes or [config.model.dtype]
    resolutions = args.resolutions or [parse_resolution(r) for r in DEFAULT_RESOLUTIONS]
    frame_size = (args.frame_size[0], args.frame_size[1])

    sections: list[str] = [
        "# Phase A: RADIO image encoder verification",
        "",
        "Generated by `scripts/verify_phase_a.py`. Latency and memory numbers",
        "describe this machine, not a performance commitment, and the image-text",
        "ranking below is a wiring check rather than an accuracy result.",
        "",
        "`--dtypes` varies the image backbone dtype only; the text tower uses",
        "`text_encoder_dtype` from the config throughout.",
        "",
        "## Environment",
        "",
        *environment_section(),
        "",
    ]

    failures: list[str] = []
    measurements: list[dict] = []
    text_measurements: list[dict] = []
    reference: dict[str, dict[tuple, np.ndarray]] = {}

    for dtype in dtypes:
        run_config = Config.from_dict({"model": {**_as_dict(config.model), "dtype": dtype}})
        started = time.monotonic()
        encoder = AlignedEncoder(run_config)
        load_seconds = time.monotonic() - started

        probes = distinct_frames(frame_size)
        contract, contract_failures = contract_section(encoder, synthetic_frame(1, frame_size))
        failures.extend(f"{dtype}: {name}" for name in contract_failures)

        alignment_lines, alignment_failures = text_alignment(encoder, frame_size)
        failures.extend(f"{dtype}: {name}" for name in alignment_failures)
        alignment_lines = [*alignment_lines, ""]
        latency = text_latency(encoder, args.runs, args.warmup)
        photo_lines = real_photo_alignment(encoder)
        text_measurements.append({"dtype": dtype, **latency})
        print(
            f"  {dtype:9} text: batch10 {latency['batch_10_ms']:7.1f} ms  "
            f"single {latency['single_ms']:6.1f} ms  device={latency['device']}",
            flush=True,
        )

        sections.extend(
            [
                f"## Loaded model (dtype `{dtype}`)",
                "",
                *load_section(encoder, load_seconds),
                "### Vector contract (spec 2.2)",
                "",
                *contract,
                "",
                "### Image-text ranking (spec 2.2 wiring check)",
                "",
                "Frames whose content is known by construction. The top label must be the",
                "diagonal cell; if it is not, the image summary is not in the adaptor's",
                "text space and nothing downstream can work.",
                "",
                *alignment_lines,
                "### Text encoding cost",
                "",
                f"Text tower on `{latency['device']}`: 10-word batch {latency['batch_10_ms']} ms "
                f"({latency['batch_10_ms'] / 10:.1f} ms per word), single word {latency['single_ms']} ms.",
                "",
                "### Similarity between six structurally distinct frames",
                "",
                "Solid red, solid blue, white, checkerboard, gradient, noise. A set of",
                "*similar* frames would collapse towards cosine 1 in any CLIP space; these",
                "are the cases a working encoder still has to separate.",
                "",
                f"```json\n{json.dumps(separation(encoder, probes), indent=2)}\n```",
                "",
                *photo_lines,
            ]
        )

        rows = []
        for resolution in resolutions:
            encoder.set_input_resolution(resolution)
            measured = measure(encoder, probes[0], args.runs, args.warmup)
            measured["dtype"] = dtype
            measurements.append(measured)
            rows.append(measured)
            reference.setdefault(dtype, {})[tuple(measured["resolution"])] = encoder.encode_images(probes)
            print(
                f"  {dtype:9} resolution={str(resolution):12} "
                f"input={measured['model_input_size']} tokens={measured['tokens']:5} "
                f"p50={measured['ms_p50']:8.1f} ms  p95={measured['ms_p95']:8.1f} ms  "
                f"peak_alloc={measured.get('peak_allocated_mb', float('nan')):.0f} MB",
                flush=True,
            )

        sections.extend(
            [
                "### Cost per input resolution (one 576x1024 frame at a time)",
                "",
                "`peak alloc` is this measurement's high-water allocation; `reserved` is the",
                "process-wide caching-allocator high-water mark, which never shrinks.",
                "",
                "| config resolution | model input | tokens | p50 ms | p95 ms | img/s | peak alloc MB | reserved MB |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            sections.append(
                f"| {row['resolution']} | {row['model_input_size'][0]}x{row['model_input_size'][1]} | "
                f"{row['tokens']} | {row['ms_p50']} | {row['ms_p95']} | {row['images_per_second']} | "
                f"{row.get('peak_allocated_mb', '')} | {row.get('peak_reserved_mb', '')} |"
            )
        sections.append("")

    if len(dtypes) > 1:
        sections.extend(
            [
                "## Agreement between dtypes",
                "",
                "Cosine between the same frames encoded in the first dtype and each later",
                "one. This is the evidence for allowing a lower-precision run; it is not a",
                "retrieval-accuracy result.",
                "",
                "| resolution | dtype pair | min cosine | mean cosine |",
                "| --- | --- | --- | --- |",
            ]
        )
        for resolution in resolutions:
            key = tuple(resolution)
            base = reference[dtypes[0]].get(key)
            if base is None:
                continue
            for dtype in dtypes[1:]:
                other = reference[dtype].get(key)
                if other is None:
                    continue
                cosines = np.sum(base * other, axis=1)
                sections.append(
                    f"| {resolution} | {dtypes[0]} vs {dtype} | "
                    f"{float(cosines.min()):.5f} | {float(cosines.mean()):.5f} |"
                )
        sections.append("")

    sections.extend(
        [
            "## Raw measurements",
            "",
            f"```json\n{json.dumps(measurements, indent=2)}\n```",
            "",
            "## Verdict",
            "",
        ]
    )
    if failures:
        sections.append("FAILED checks:")
        sections.extend(f"- {name}" for name in failures)
    else:
        sections.append("All hard checks passed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"\nreport written to {args.out}")

    if failures:
        print("FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def _as_dict(model_config) -> dict:
    import dataclasses

    return {
        field.name: getattr(model_config, field.name)
        for field in dataclasses.fields(model_config)
        if field.name != "checkpoint_path"
    }


if __name__ == "__main__":
    raise SystemExit(main())
