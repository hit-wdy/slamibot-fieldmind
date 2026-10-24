"""Record live camera frames: YOLO detections + RADIO embedding + pose, at 1 Hz.

YOLO runs on the OAK VPU and RADIO on the host GPU, so they genuinely run at the
same time rather than taking turns. No images are written: a record holds the
frame identity, the embedding, the detections and the camera pose.

Pose is simulated until the SLAM system is wired in. Every record carries
`pose_source` so a simulated pose can never be read as a measured one.

Storage is SQLite plus float16 vector partitions (spec section 6). The collection
refuses to accept vectors from a different model fingerprint.

The camera is opened *before* the model loads. That claims the device as early as
possible on hardware whose USB link is unreliable, and auto-exposure settles
during the load rather than after it.

Examples:
    python scripts/record_live.py --seconds 60
    python scripts/record_live.py --seconds 60 --interval 1.0 --out data/records
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from slamibot_fieldmind import AlignedEncoder, Config
from slamibot_fieldmind.capture import CameraSource
from slamibot_fieldmind.detect import YoloV6Detector, summarise
from slamibot_fieldmind.paths import PROJECT_ROOT
from slamibot_fieldmind.pose import SimulatedPoseSource
from slamibot_fieldmind.sampling import IntervalSampler
from slamibot_fieldmind.store import RecordStore

DEFAULT_OUT = PROJECT_ROOT / "data" / "records"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to record")
    parser.add_argument("--interval", type=float, default=1.0, help="index interval in seconds")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="record root")
    parser.add_argument("--collection-id", default="radio_clip_v1")
    parser.add_argument("--camera-id", default="camera_01")
    parser.add_argument("--location-label", default=None, help="human-readable camera place")
    parser.add_argument("--device", default=None)
    parser.add_argument("--socket", default="CAM_A")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 576], metavar=("W", "H"))
    parser.add_argument("--fps", type=float, default=10.0, help="camera rate; also the YOLO rate")
    parser.add_argument("--roi-inset", type=float, default=0.15, help="fisheye border dropped before YOLO")
    parser.add_argument("--confidence", type=float, default=0.25, help="native objectness*class candidate threshold; precision policy then applies per-class thresholds")
    parser.add_argument("--no-settle", action="store_true", help="skip waiting for auto-exposure")
    parser.add_argument("--pose-seed", type=int, default=0, help="seed for the simulated pose trajectory")
    parser.add_argument("--pose-period", type=float, default=60.0, help="seconds per lap of the simulated path")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser.parse_args(argv)


class _Stopper:
    def __init__(self) -> None:
        self.requested = False

    def __call__(self, _signum: int, _frame: object) -> None:
        self.requested = True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    detector = YoloV6Detector(roi_inset=args.roi_inset, confidence_threshold=args.confidence)
    pose_source = SimulatedPoseSource(seed=args.pose_seed, period_s=args.pose_period)
    sampler = IntervalSampler(interval_s=args.interval)
    stopper = _Stopper()
    signal.signal(signal.SIGINT, stopper)

    camera = CameraSource(
        camera_id=args.camera_id,
        socket=args.socket,
        resolution=(args.resolution[0], args.resolution[1]),
        fps=args.fps,
        device_id=args.device,
        detector=detector,
        settle=not args.no_settle,
    )

    with camera:
        print(f"camera: {json.dumps(camera.device_info)}", flush=True)
        print(f"detector: {json.dumps(camera.detector_info)}", flush=True)
        settle = camera.settle_info
        if settle:
            print(
                f"exposure: {'settled' if settle.get('settled') else 'NOT settled'} after "
                f"{settle['frames']} frames / {settle['seconds']}s "
                f"(brightness {settle['brightness']}, {100 * settle['saturated_fraction']:.2f}% saturated)",
                flush=True,
            )

        print("loading RADIO...", flush=True)
        started = time.monotonic()
        encoder = AlignedEncoder(Config.default())
        print(
            f"  {encoder.load_report.version} [{encoder.load_report.adaptor}] dim={encoder.embedding_dim} "
            f"dtype={encoder.load_report.model_dtype} fingerprint={encoder.fingerprint_short()} "
            f"({time.monotonic() - started:.1f}s)",
            flush=True,
        )
        print(f"flushed {camera.flush()} frames that arrived during the load", flush=True)

        configuration = {
            "adaptor": encoder.load_report.adaptor,
            "model_version": encoder.load_report.version,
            "run_dtype": encoder.load_report.model_dtype,
            "input_resolution": list(encoder.config.model.input_resolution),
            "index_interval_s": args.interval,
            "detector": detector.describe(),
            "pose": pose_source.describe(),
            "stores_images": False,
        }
        with RecordStore(
            root=args.out,
            collection_id=args.collection_id,
            model_fingerprint=encoder.fingerprint(),
            embedding_dim=encoder.embedding_dim,
            configuration=configuration,
        ) as store:
            store.register_camera(
                args.camera_id,
                source_type="oak_camera",
                location_label=args.location_label,
                pose_frame=pose_source.frame,
                pose_source=pose_source.describe()["source"],
            )
            print(
                f"collection {args.collection_id} at {store.db_path} (existing frames: {store.count()})",
                flush=True,
            )

            written = 0
            encode_ms: list[float] = []
            motion: list[float] = []
            wall_start = time.monotonic()
            last_pose = None
            next_report = wall_start + 5.0

            while not stopper.requested and time.monotonic() - wall_start < args.seconds:
                frame = camera.grab()
                if not sampler.observe():
                    continue

                elapsed = time.monotonic() - wall_start
                encode_started = time.monotonic()
                embedding = encoder.encode_images([frame.image])[0]
                encode_ms.append((time.monotonic() - encode_started) * 1000.0)

                pose = pose_source.at(elapsed)
                if last_pose is not None:
                    motion.append(last_pose.distance_to(pose))
                last_pose = pose

                record = store.add(
                    camera_id=frame.camera_id,
                    captured_at=frame.captured_at,
                    source_sequence=frame.source_sequence,
                    embedding=embedding,
                    pose=pose,
                    detections=frame.detections,
                    timestamp_source=frame.timestamp_source,
                    device_timestamp_ns=frame.device_timestamp_ns,
                )
                written += 1

                if not frame.detections_available:
                    print(f"  WARNING: no YOLO result for {record.frame_id}; stored zero detections", flush=True)
                if frame.saturated_fraction > 0.2:
                    print(
                        f"  WARNING: {record.frame_id} is {100 * frame.saturated_fraction:.1f}% saturated",
                        flush=True,
                    )

                counts = summarise(frame.detections)
                print(
                    f"  [{written:3d}] {record.frame_id} dets={len(frame.detections)} "
                    f"{json.dumps(counts) if counts else ''} "
                    f"pose=({pose.x:+.2f},{pose.y:+.2f},{pose.z:+.2f}) encode={encode_ms[-1]:.0f}ms",
                    flush=True,
                )

                now = time.monotonic()
                if now >= next_report:
                    print(
                        f"  ... {written} records, {sampler.stats.observed_hz:.2f} Hz observed, "
                        f"encode p50 {np.median(encode_ms):.0f}ms, {now - wall_start:.0f}s elapsed",
                        flush=True,
                    )
                    next_report = now + 5.0

            summary = {
                "collection_id": args.collection_id,
                "frames_written": written,
                "frames_in_collection": store.count(),
                "sampling": sampler.stats.as_dict(),
                "encode_ms_p50": round(float(np.median(encode_ms)), 1) if encode_ms else None,
                "encode_ms_p95": round(float(np.percentile(encode_ms, 95)), 1) if encode_ms else None,
                "camera_motion_total_m": round(float(sum(motion)), 3),
                "detector": detector.describe(),
                "pose_source": pose_source.describe()["source"],
                "store": store.stats(),
                "integrity": store.verify(),
            }

    print()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"records written: {summary['frames_written']} (collection now {summary['frames_in_collection']})")
        print(f"  sampling: {json.dumps(summary['sampling'])}")
        print(f"  encode: p50 {summary['encode_ms_p50']}ms  p95 {summary['encode_ms_p95']}ms")
        print(f"  simulated camera travel: {summary['camera_motion_total_m']} m")
        print(f"  store: {json.dumps(summary['store'], ensure_ascii=False)}")
        integrity = summary["integrity"]
        print(f"  integrity: {'OK' if integrity['ok'] else integrity['orphan_or_missing']}")
    return 0 if summary["integrity"]["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, TimeoutError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
