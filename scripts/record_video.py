"""Record a video clip from the camera, optionally alongside the 1 Hz index.

Why this exists: `record_live.py` stores no images by design, so a retrieval
result cannot be checked against what the camera actually saw. This script keeps
the picture and, crucially, records which video frame each indexed record came
from, so `frame_id -> seek the video to this second` is a direct lookup.

    data/captures/<UTC start>/
      video.mp4        every frame at the camera rate
      manifest.json    camera info, timing, and the record -> video frame map
      indexed/         the 1 Hz sampled frames as JPEG, named by frame_id

`--no-index` skips the encoder and the store entirely: no 25 s model load, no
YOLO, just a video. The camera can only be held by one process at a time, which
is why this is one script rather than two run side by side.

Examples:
    python scripts/record_video.py --seconds 60
    python scripts/record_video.py --seconds 60 --no-index
    python scripts/record_video.py --seconds 120 --fps 10 --location-label "厂区东门"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from slamibot_fieldmind.capture import CameraSource
from slamibot_fieldmind.detect import YoloV6Detector, summarise
from slamibot_fieldmind.paths import PROJECT_ROOT
from slamibot_fieldmind.pose import SimulatedPoseSource
from slamibot_fieldmind.sampling import IntervalSampler

DEFAULT_OUT = PROJECT_ROOT / "data" / "captures"
DEFAULT_RECORDS = PROJECT_ROOT / "data" / "records"
FOURCC = "mp4v"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60.0, help="clip length")
    parser.add_argument("--fps", type=float, default=10.0, help="camera and video rate")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="capture root")
    parser.add_argument("--capture-id", default=None, help="folder name; defaults to the UTC start time")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 576], metavar=("W", "H"))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--usb-speed",
        choices=("AUTO", "HIGH", "SUPER"),
        default="AUTO",
        help="use HIGH when the device fails to boot over USB3 (X_LINK_DEVICE_NOT_FOUND after booting)",
    )
    parser.add_argument("--socket", default="CAM_A")
    parser.add_argument("--no-settle", action="store_true", help="skip waiting for auto-exposure")

    parser.add_argument("--no-index", action="store_true", help="video only: no model, no detections, no records")
    parser.add_argument("--interval", type=float, default=1.0, help="index interval in seconds")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--collection-id", default="radio_clip_v1")
    parser.add_argument("--camera-id", default="camera_01")
    parser.add_argument("--location-label", default=None)
    parser.add_argument("--roi-inset", type=float, default=0.15)
    parser.add_argument("--confidence", type=float, default=0.25, help="native objectness*class threshold before the per-class precision policy")
    parser.add_argument("--no-jpeg", action="store_true", help="do not also save the indexed frames as JPEG")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--no-frame-timestamps", action="store_true", help="omit the per-frame clock list from the manifest")
    return parser.parse_args(argv)


class _Stopper:
    def __init__(self) -> None:
        self.requested = False

    def __call__(self, _signum: int, _frame: object) -> None:
        self.requested = True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    width, height = args.resolution
    capture_id = args.capture_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.out / capture_id
    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / "video.mp4"
    indexed_dir = directory / "indexed"
    if not args.no_index and not args.no_jpeg:
        indexed_dir.mkdir(parents=True, exist_ok=True)

    stopper = _Stopper()
    signal.signal(signal.SIGINT, stopper)

    detector = None if args.no_index else YoloV6Detector(
        roi_inset=args.roi_inset, confidence_threshold=args.confidence
    )
    sampler = IntervalSampler(interval_s=args.interval)
    pose_source = SimulatedPoseSource()

    # Claim the device before anything slow, and let auto-exposure settle while
    # the model loads rather than after.
    camera = CameraSource(
        camera_id=args.camera_id,
        socket=args.socket,
        resolution=(width, height),
        fps=args.fps,
        device_id=args.device,
        usb_speed=args.usb_speed,
        detector=detector,
        queue_size=8,
        settle=not args.no_settle,
    )

    writer: cv2.VideoWriter | None = None
    indexed: list[dict] = []
    frame_clocks: list[int] = []
    encode_ms: list[float] = []

    def cue(text: str = "") -> None:
        print(">>> " + text if text else ">>>", flush=True)

    with camera:
        print(f"camera: {json.dumps(camera.device_info)}", flush=True)
        if detector is not None:
            print(f"detector: {json.dumps(camera.detector_info)}", flush=True)
        settle = camera.settle_info
        if settle:
            print(
                f"exposure: {'settled' if settle.get('settled') else 'NOT settled'} after "
                f"{settle['frames']} frames / {settle['seconds']}s "
                f"(brightness {settle['brightness']}, {100 * settle['saturated_fraction']:.2f}% saturated)",
                flush=True,
            )

        encoder = None
        store = None
        if not args.no_index:
            print(f"[2/3] 加载模型中，约 25 秒 —— 这段时间请先别动相机", flush=True)
            started = time.monotonic()
            from slamibot_fieldmind import AlignedEncoder, Config

            encoder = AlignedEncoder(Config.default())
            print(
                f"  {encoder.load_report.version} [{encoder.load_report.adaptor}] dim={encoder.embedding_dim} "
                f"dtype={encoder.load_report.model_dtype} fingerprint={encoder.fingerprint_short()} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
            from slamibot_fieldmind.store import RecordStore

            store = RecordStore(
                root=args.records_root,
                collection_id=args.collection_id,
                model_fingerprint=encoder.fingerprint(),
                embedding_dim=encoder.embedding_dim,
                configuration={
                    "adaptor": encoder.load_report.adaptor,
                    "run_dtype": encoder.load_report.model_dtype,
                    "index_interval_s": args.interval,
                    "detector": detector.describe(),
                    "pose": pose_source.describe(),
                    "stores_images": False,
                    "capture_id": capture_id,
                },
            ).open()
            store.register_camera(
                args.camera_id,
                source_type="oak_camera",
                location_label=args.location_label,
                pose_frame=pose_source.frame,
                pose_source=pose_source.describe()["source"],
            )
            print(f"  records -> {store.db_path} (existing frames: {store.count()})", flush=True)
            print(f"flushed {camera.flush()} frames that arrived during the load", flush=True)

        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*FOURCC), args.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open a video writer for {video_path} (codec {FOURCC!r})")
        print(f"      视频写入: {video_path}", flush=True)
        print(flush=True)
        cue("=" * 62)
        cue(f"  准备好了。{int(args.seconds)} 秒后开始，请先站好位置：")
        cue("=" * 62)
        for remaining in (5, 4, 3, 2, 1):
            print(f"        {remaining} ...", flush=True)
            time.sleep(1.0)
        print(flush=True)
        cue("=" * 62)
        cue(f"  ▶▶  现在开始移动相机！录制约 {int(args.seconds)} 秒。")
        cue("=" * 62)
        print(flush=True)

        written = 0
        wall_start = time.monotonic()
        next_report = wall_start + 10.0
        capture_started_at = dt.datetime.now(dt.timezone.utc)

        try:
            while not stopper.requested and time.monotonic() - wall_start < args.seconds:
                frame = camera.grab()
                keep = sampler.observe()
                if frame.device_timestamp_ns is not None:
                    frame_clocks.append(frame.device_timestamp_ns)

                writer.write(frame.image[:, :, ::-1])  # RGB -> BGR for OpenCV
                video_index = written
                written += 1

                if keep and encoder is not None and store is not None:
                    elapsed = time.monotonic() - wall_start
                    started = time.monotonic()
                    embedding = encoder.encode_images([frame.image])[0]
                    encode_ms.append((time.monotonic() - started) * 1000.0)
                    pose = pose_source.at(elapsed)
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
                    entry = {
                        "frame_id": record.frame_id,
                        "video_frame": video_index,
                        "video_seconds": round(video_index / args.fps, 3),
                        "captured_at": frame.captured_at.isoformat(timespec="milliseconds"),
                        "device_timestamp_ns": frame.device_timestamp_ns,
                        "source_sequence": frame.source_sequence,
                        "brightness": round(frame.brightness, 1),
                        "saturated_fraction": round(frame.saturated_fraction, 4),
                        "detections": [d.to_dict() for d in frame.detections],
                    }
                    if not args.no_jpeg:
                        jpeg_path = indexed_dir / f"{record.frame_id}.jpg"
                        cv2.imwrite(str(jpeg_path), frame.image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                        entry["jpeg"] = str(jpeg_path.relative_to(directory))
                    indexed.append(entry)
                    counts = summarise(frame.detections)
                    print(
                        f"  [{len(indexed):3d}] video#{video_index:5d} ({video_index / args.fps:6.1f}s) "
                        f"{record.frame_id} dets={json.dumps(counts) if counts else 0} "
                        f"encode={encode_ms[-1]:.0f}ms",
                        flush=True,
                    )

                now = time.monotonic()
                if now >= next_report:
                    left = max(0.0, args.seconds - (now - wall_start))
                    print(
                        f"  ... {written} video frames, {len(indexed)} indexed, "
                        f"{now - wall_start:.0f}s elapsed, 剩余 {left:.0f}s",
                        flush=True,
                    )
                    next_report = now + 10.0
        finally:
            if writer is not None:
                writer.release()
                writer = None
            if store is not None:
                store.close()

    elapsed = time.monotonic() - wall_start
    measured_fps = written / elapsed if elapsed > 0 else 0.0
    print(flush=True)
    cue("=" * 62)
    cue(f"  ■■  录制结束，可以停下来了。{written} 帧视频 / {len(indexed)} 条记录")
    cue("=" * 62)
    manifest = {
        "capture_id": capture_id,
        "started_at": capture_started_at.isoformat(timespec="milliseconds"),
        "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "elapsed_s": round(elapsed, 2),
        "camera": camera.device_info,
        "video": {
            "path": video_path.name,
            "codec": FOURCC,
            "declared_fps": args.fps,
            "measured_fps": round(measured_fps, 3),
            "width": width,
            "height": height,
            "frames": written,
            "note": (
                "frames are written at the declared fps; if measured_fps differs the clock drifts. "
                "per_frame_device_timestamp_ns gives the true ordering."
            ),
        },
        "indexing": {
            "enabled": not args.no_index,
            "interval_s": args.interval,
            "collection_id": None if args.no_index else args.collection_id,
            "records_root": None if args.no_index else str(args.records_root),
            "frames_indexed": len(indexed),
            "observed_hz": round(sampler.stats.observed_hz, 3),
            "sampler": sampler.stats.as_dict(),
            "encode_ms_p50": round(float(np.median(encode_ms)), 1) if encode_ms else None,
            "encode_ms_p95": round(float(np.percentile(encode_ms, 95)), 1) if encode_ms else None,
        },
        "pose_source": pose_source.describe()["source"],
        "pose_note": (
            "poses stored alongside these records are simulated; they do not describe where this "
            "clip was filmed"
        ),
        "how_to_debug": (
            "search_frames returns a frame_id; find it under indexing.indexed to get video_frame "
            "and video_seconds, then seek video.mp4 there. The JPEG next to it, if saved, is the "
            "exact image that was embedded."
        ),
        "indexed": indexed,
    }
    if not args.no_frame_timestamps:
        manifest["video"]["per_frame_device_timestamp_ns"] = frame_clocks

    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"capture  : {directory}")
    print(f"video    : {written} frames, {elapsed:.1f}s, measured {measured_fps:.2f} fps")
    print(f"indexed  : {len(indexed)} records at {sampler.stats.observed_hz:.2f} Hz")
    if encode_ms:
        print(f"encode   : p50 {np.median(encode_ms):.0f}ms  p95 {np.percentile(encode_ms, 95):.0f}ms")
    print(f"manifest : {manifest_path}")
    if measured_fps < args.fps * 0.95:
        print(
            f"WARNING: measured {measured_fps:.2f} fps against a declared {args.fps:.0f}; the video clock "
            "drifts, use per_frame_device_timestamp_ns to place a frame precisely"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, TimeoutError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
