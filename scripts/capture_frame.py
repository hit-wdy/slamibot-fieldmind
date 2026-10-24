"""Capture frames from the OAK camera (spec section 4 input adapter).

Prints per-frame exposure diagnostics, because a blown-out frame produces an
embedding that describes nothing and is hard to notice downstream.

Examples:
    python scripts/capture_frame.py --count 1 --out data/live/frame.jpg
    python scripts/capture_frame.py --count 10 --out-dir data/live/session1
    python scripts/capture_frame.py --manual-exposure 2000 200   # diagnose auto-exposure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from slamibot_fieldmind.capture import CameraSource, list_devices
from slamibot_fieldmind.paths import PROJECT_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1, help="how many frames to save")
    parser.add_argument("--out", type=Path, default=None, help="write a single frame here")
    parser.add_argument("--out-dir", type=Path, default=None, help="write every frame here")
    parser.add_argument("--camera-id", default="camera_01")
    parser.add_argument("--device", default=None, help="DepthAI device id (needed if several are attached)")
    parser.add_argument("--socket", default="CAM_A")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 576], metavar=("W", "H"))
    parser.add_argument("--fps", type=float, default=None, help="camera rate; default is the device's own")
    parser.add_argument("--no-settle", action="store_true", help="do not wait for auto-exposure to converge")
    parser.add_argument("--settle-timeout", type=float, default=15.0, help="give up settling after this many seconds")
    parser.add_argument(
        "--manual-exposure",
        type=int,
        nargs=2,
        default=None,
        metavar=("US", "ISO"),
        help="pin exposure instead of auto; the way to tell a broken AE loop from a bright scene",
    )
    parser.add_argument("--max-exposure-us", type=int, default=None, help="cap auto-exposure")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality")
    parser.add_argument("--list", action="store_true", help="list devices and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        devices = list_devices()
        print(json.dumps({"depthai_devices": devices}, indent=2))
        return 0 if devices else 2

    if args.out is None and args.out_dir is None:
        args.out = PROJECT_ROOT / "data" / "live" / "frame.jpg"
    if args.out is not None and args.count != 1:
        raise SystemExit("--out takes a single frame; use --out-dir for several")

    exposure_us, iso = (args.manual_exposure if args.manual_exposure else (None, None))
    source = CameraSource(
        camera_id=args.camera_id,
        socket=args.socket,
        resolution=(args.resolution[0], args.resolution[1]),
        fps=args.fps,
        device_id=args.device,
        settle=not args.no_settle,
        settle_timeout=args.settle_timeout,
        exposure_us=exposure_us,
        iso=iso,
        auto_exposure_limit_us=args.max_exposure_us,
    )

    with source:
        print(f"device: {json.dumps(source.device_info)}")
        print(f"exposure settle: {json.dumps(source.settle_info)}")
        if source.settle_info and not source.settle_info.get("settled"):
            print("       WARNING: " + str(source.settle_info.get("reason")))
        if exposure_us or iso:
            print(f"exposure pinned to {exposure_us or 'auto'} us / iso {iso or 'auto'}")
        written = []
        for index, frame in enumerate(source.frames(count=args.count)):
            exposure = f"{frame.exposure_time_us} us" if frame.exposure_time_us else "?"
            iso_text = f"iso {frame.iso}" if frame.iso else "iso ?"
            print(
                f"  [{index}] seq={frame.source_sequence} {frame.width}x{frame.height} "
                f"brightness={frame.brightness:6.1f} saturated={100 * frame.saturated_fraction:5.1f}% "
                f"exposure={exposure} {iso_text} captured_at={frame.captured_at.isoformat(timespec='milliseconds')}"
            )
            if frame.saturated_fraction > 0.2:
                print(
                    "       WARNING: over 20% of this frame is fully saturated; an embedding from it "
                    "describes the clipping, not the scene."
                )
            if args.out is not None:
                path = args.out
            else:
                path = args.out_dir / f"{frame.captured_at.strftime('%Y%m%dT%H%M%S')}_{frame.source_sequence:06d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.image).save(path, quality=args.quality)
            written.append(str(path))

        print(json.dumps({"frames": written}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, TimeoutError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
