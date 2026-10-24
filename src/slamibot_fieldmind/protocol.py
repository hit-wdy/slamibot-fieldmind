"""Version 1 wire validation for the ROS-independent backend."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import re

from PIL import Image

from .keyframes import pose_dict

MAX_JPEG_BYTES = 16 * 1024 * 1024
UTC = dt.timezone.utc


def utcnow():
    return dt.datetime.now(UTC).isoformat(timespec="milliseconds")


def identifier(value, field):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value)
        or value in {".", ".."}
    ):
        raise ValueError(f"invalid {field}")
    return value


def timestamp(value):
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamps require a timezone offset")
    return parsed.astimezone(UTC).isoformat(timespec="milliseconds")


def validate_observation(raw, jpeg):
    m = dict(raw)
    if m.get("protocol_version", 1) != 1:
        raise ValueError("unsupported protocol version")
    for k in ("camera_id", "device_id", "source_id"):
        m[k] = identifier(m.get(k), k)
    for k in ("source_sequence", "source_stamp_ns"):
        if isinstance(m.get(k), bool) or not isinstance(m.get(k), int) or m[k] < 0:
            raise ValueError(f"invalid {k}")
    for k in ("session_id", "coordinate_id"):
        if m.get(k) is not None:
            identifier(m[k], k)
    m["received_at"] = timestamp(m["received_at"])
    # UTC acquisition time is optional: ROS/device clocks need not be Unix time.
    m["captured_at"] = timestamp(m.get("captured_at") or m["received_at"])
    m["timestamp_source"] = str(m.get("timestamp_source", "unknown"))
    m["sync_quality"] = str(m.get("sync_quality", "unknown"))
    m["body_frame_id"] = (m.get("body_pose") or {}).get("child_frame")
    m["body_pose"] = pose_dict(m.get("body_pose"))
    m["camera_pose"] = pose_dict(m.get("camera_pose"))
    if m["camera_pose"] is not None and not m.get("coordinate_id"):
        raise ValueError("camera_pose requires a coordinate_id")
    if m.get("detection_status") not in {"available", "unavailable"}:
        raise ValueError("detection_status must be available or unavailable")
    detections = m.get("detections", [])
    if not isinstance(detections, list) or len(detections) > 300:
        raise ValueError("invalid detections")
    clean = []
    for d in detections:
        box = [float(v) for v in d["bbox_norm"]]
        conf = float(d["confidence"])
        if (
            len(box) != 4
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in box)
            or box[0] > box[2]
            or box[1] > box[3]
        ):
            raise ValueError("bbox_norm must be full-image xyxy in [0,1]")
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("invalid confidence")
        clean.append(
            dict(
                label=str(d["label"]),
                label_id=int(d["label_id"]),
                confidence=conf,
                bbox_norm=box,
            )
        )
    if m["detection_status"] != "available" and clean:
        raise ValueError("unavailable detections cannot contain boxes")
    m["detections"] = clean
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError("JPEG is empty or exceeds 16 MiB")
    with Image.open(io.BytesIO(jpeg)) as image:
        if image.format != "JPEG" or image.width * image.height > 24_000_000:
            raise ValueError("expected JPEG of at most 24 megapixels")
        image.verify()
        m["image_width"], m["image_height"] = image.size
    # Include the capture producer's boot ID; device sequence counters restart.
    key = [
        m.get("session_id"),
        m.get("coordinate_id"),
        m["device_id"],
        m["camera_id"],
        m["source_id"],
        m["source_sequence"],
    ]
    m["frame_id"] = "kf-" + hashlib.sha256(repr(key).encode()).hexdigest()[:32]
    return m
