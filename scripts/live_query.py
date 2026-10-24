"""Live camera -> RADIO embedding -> text ranking.

This is an integration demo, not the retrieval service. It holds the vectors in
memory for the length of one run: there is no index, no persistence, and no time
or camera filtering, so nothing survives the process. Spec sections 5-8 are still
unimplemented.

Its other job is to make the two failure modes visible:

* a blown-out frame still yields a unit-norm embedding, so exposure is reported
  and warned about rather than silently trusted;
* consecutive frames from a static scene are near-duplicates (measured: pairwise
  cosine 0.9985-0.9998 over one second), so the within-query spread is printed
  next to the top score. Without time grouping, top-K fills up with one moment.

Examples:
    python scripts/live_query.py --queries "a computer monitor" "a fire truck"
    python scripts/live_query.py --frames 20 --interval 1.0 --queries "a person" --out-dir data/live/run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from slamibot_fieldmind import AlignedEncoder, Config
from slamibot_fieldmind.capture import CameraSource

DEFAULT_QUERIES = ["a computer monitor", "a speaker", "a cat", "a fire truck"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES, help="English visual phrases to rank")
    parser.add_argument("--frames", type=int, default=10, help="how many frames to capture")
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="seconds between captured frames; default is back-to-back",
    )
    parser.add_argument("--top-k", type=int, default=2, help="frames to show per query")
    parser.add_argument("--camera-id", default="camera_01")
    parser.add_argument("--device", default=None)
    parser.add_argument("--socket", default="CAM_A")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 576], metavar=("W", "H"))
    parser.add_argument("--fps", type=float, default=10.0, help="camera rate")
    parser.add_argument("--no-settle", action="store_true", help="skip waiting for auto-exposure")
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--out-dir", type=Path, default=None, help="save the captured frames here")
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--json", action="store_true", help="also emit a machine-readable result")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_k > args.frames:
        args.top_k = args.frames

    started = time.monotonic()
    encoder = AlignedEncoder(Config.default())
    load_seconds = time.monotonic() - started
    print(f"encoder: {encoder.load_report.version} [{encoder.load_report.adaptor}] "
          f"dim={encoder.embedding_dim} dtype={encoder.load_report.model_dtype} "
          f"fingerprint={encoder.fingerprint_short()} ({load_seconds:.1f}s)")

    with CameraSource(
        camera_id=args.camera_id,
        socket=args.socket,
        resolution=(args.resolution[0], args.resolution[1]),
        fps=args.fps,
        device_id=args.device,
        settle=not args.no_settle,
        settle_timeout=args.settle_timeout,
    ) as camera:
        settle = camera.settle_info
        print(f"camera: {camera.device_info['platform']} usb={camera.device_info['usb_speed']}")
        if settle:
            verdict = "settled" if settle.get("settled") else "NOT settled"
            print(f"exposure: {verdict} after {settle['frames']} frames / {settle['seconds']}s "
                  f"(brightness {settle['brightness']}, {100 * settle['saturated_fraction']:.2f}% saturated)")
            if not settle.get("settled"):
                print(f"  WARNING: {settle.get('reason')}")

        frames = []
        for index in range(args.frames):
            frames.append(camera.grab())
            if args.interval and index + 1 < args.frames:
                time.sleep(args.interval)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            name = f"{frame.captured_at.strftime('%Y%m%dT%H%M%S%f')[:-3]}_{frame.source_sequence:06d}.jpg"
            Image.fromarray(frame.image).save(args.out_dir / name, quality=args.quality)

    hot = [f for f in frames if f.saturated_fraction > 0.2]
    brightness = [f.brightness for f in frames]
    print(f"frames: {len(frames)}  brightness {min(brightness):.1f}-{max(brightness):.1f}  "
          f"max saturated {100 * max(f.saturated_fraction for f in frames):.2f}%")
    if hot:
        print(f"  WARNING: {len(hot)} of {len(frames)} frames are over 20% saturated; "
              "the embeddings describe clipping, not the scene.")
    if frames[0].timestamp_source != "camera_system_clock":
        print(f"  note: captured_at falls back to host receive time ({frames[0].timestamp_source}); "
              "the device clock is a monotonic duration, not wall time")

    started = time.monotonic()
    images = encoder.encode_images([frame.image for frame in frames])
    encode_seconds = time.monotonic() - started

    similarity = images @ images.T
    off_diagonal = similarity[~np.eye(len(frames), dtype=bool)]
    print(f"encode: {1000 * encode_seconds / len(frames):.1f} ms/frame  "
          f"pairwise cosine {off_diagonal.min():.4f}-{off_diagonal.max():.4f}")
    if off_diagonal.max() > 0.99:
        print("  note: frames are near-duplicates; without time grouping these would all occupy top-K")

    results = {}
    print()
    for query in args.queries:
        scores = images @ encoder.encode_texts([query])[0]
        order = np.argsort(-scores)[: args.top_k]
        results[query] = [
            {
                "frame_id": f"{frames[i].camera_id}:{frames[i].source_sequence}",
                "similarity": round(float(scores[i]), 4),
                "captured_at": frames[i].captured_at.isoformat(timespec="milliseconds"),
                "brightness": round(frames[i].brightness, 1),
            }
            for i in order
        ]
        top = results[query][0]
        print(f"{query!r}")
        print(f"   top {top['similarity']:+.4f}  seq={top['frame_id']}  {top['captured_at']}")
        print(f"   this-run spread {scores.min():+.4f} .. {scores.max():+.4f}  "
              f"(a negative or near-zero top score is not evidence the object is present)")

    if args.json:
        print()
        print(json.dumps(
            {
                "fingerprint": encoder.fingerprint(),
                "embedding_dim": encoder.embedding_dim,
                "frames_captured": len(frames),
                "ms_per_frame": round(1000 * encode_seconds / len(frames), 2),
                "pairwise_cosine": {
                    "min": round(float(off_diagonal.min()), 5),
                    "max": round(float(off_diagonal.max()), 5),
                },
                "results": results,
            },
            indent=2,
        ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, TimeoutError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
