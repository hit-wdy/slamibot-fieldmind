"""Embed one or more images with RADIO and report the result.

Step 1 of RADIO_Camera_Text_Retrieval_Spec.md. This is the wiring check: it
turns images into L2-normalised float32 vectors in the CLIP-aligned RADIO space
and prints enough evidence to confirm the encoder is doing what the spec
requires. It is not a retrieval API and not an accuracy claim.

Examples:
    python scripts/embed_image.py frame.jpg
    python scripts/embed_image.py frames/*.jpg --out embeddings.npy
    python scripts/embed_image.py frame.jpg --resolution 512 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from slamibot_fieldmind import AlignedEncoder, Config, describe
from slamibot_fieldmind.preprocess import resize_size, to_rgb_pil


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="image files to embed")
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: configs/default.yaml)")
    parser.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="input resolution: one value keeps the aspect ratio (shortest edge), two centre-crop",
    )
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default=None)
    parser.add_argument("--device", default=None, help="torch device for the image backbone, e.g. cuda:0 or cpu")
    parser.add_argument("--text-encoder-device", default=None, help="torch device for the CLIP text tower")
    parser.add_argument(
        "--channel-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="channel order of the input images (bgr is what DepthAI publishes by default)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the [B, D] float32 matrix here as .npy")
    parser.add_argument("--no-verify-checkpoint", action="store_true", help="skip the checkpoint SHA-256 check")
    parser.add_argument("--quiet", action="store_true", help="only print the JSON result")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    config = Config.from_yaml(args.config) if args.config else Config.default()
    if args.resolution:
        config.model.input_resolution = args.resolution
    if args.dtype:
        config.model.dtype = args.dtype
    if args.device:
        config.model.device = args.device
    if args.text_encoder_device:
        config.model.text_encoder_device = args.text_encoder_device
    # Re-run the dataclass validation against the overrides.
    config.model.__post_init__()
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [str(p) for p in args.images if not p.is_file()]
    if missing:
        raise SystemExit(f"not a file: {', '.join(missing)}")

    config = build_config(args)
    if not args.quiet:
        print(f"loading {config.model.version} [{config.model.adaptor}] ...", flush=True)

    started = time.monotonic()
    encoder = AlignedEncoder(config, verify_checkpoint=not args.no_verify_checkpoint)
    load_seconds = time.monotonic() - started

    if not args.quiet:
        print(describe(encoder.load_report, encoder))
        print()

    prepared_sizes = []
    for path in args.images:
        pil = to_rgb_pil(path)
        height, width = resize_size(
            pil.height,
            pil.width,
            config.model.input_resolution,
            encoder.resize_multiple,
            max_dim=config.model.max_dim,
        )
        if len(config.model.input_resolution) == 2:
            height, width = config.model.input_resolution
        prepared_sizes.append([int(height), int(width)])

    started = time.monotonic()
    embeddings = encoder.encode_images(
        [str(p) for p in args.images],
        channel_order=args.channel_order,
    )
    elapsed = time.monotonic() - started

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, embeddings)

    result = {
        "fingerprint": encoder.fingerprint(),
        "fingerprint_short": encoder.fingerprint_short(),
        "embedding_dim": encoder.embedding_dim,
        "count": int(embeddings.shape[0]),
        "load_seconds": round(load_seconds, 3),
        "encode_seconds": round(elapsed, 4),
        "encode_ms_per_image": round(1000.0 * elapsed / len(args.images), 2),
        "out": str(args.out) if args.out else None,
        "environment": encoder.environment,
        "images": [
            {
                "path": str(path),
                "model_input_size": prepared_sizes[i],
                "norm": round(float(np.linalg.norm(embeddings[i])), 6),
                "checksum": hashlib.sha256(embeddings[i].tobytes()).hexdigest()[:16],
            }
            for i, path in enumerate(args.images)
        ],
    }
    if len(embeddings) > 1:
        similarity = embeddings @ embeddings.T
        result["pairwise_cosine"] = {
            "min": round(float(similarity.min()), 6),
            "max_off_diagonal": round(
                float(similarity[~np.eye(len(embeddings), dtype=bool)].max()), 6
            ),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
