"""Deterministic spatial/temporal sampling; no camera, ROS or model imports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("quaternion must contain four finite, nonzero values (xyzw)")
    return q / np.linalg.norm(q)


def rotation_deg(a, b):
    return math.degrees(
        2 * math.acos(float(np.clip(abs(quaternion(a) @ quaternion(b)), 0, 1)))
    )


def pose_dict(value):
    if value is None:
        return None
    p = {k: float(value[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    if not all(math.isfinite(v) for v in p.values()):
        raise ValueError("pose must be finite")
    q = quaternion([p[k] for k in ("qx", "qy", "qz", "qw")])
    p.update(zip(("qx", "qy", "qz", "qw"), q.tolist()))
    p["frame"] = str(value["frame"])
    p["source"] = str(value.get("source", "unknown"))
    return p


@dataclass(frozen=True)
class KeyframePolicy:
    distance_m: float = 0.5
    rotation_deg: float = 15.0
    max_interval_s: float = 3.0
    min_interval_s: float = 0.5
    timeout_distance_m: float = 0.05
    timeout_rotation_deg: float = 2.0

    def __post_init__(self):
        if (
            any(
                not math.isfinite(x) or x <= 0
                for x in (
                    self.distance_m,
                    self.rotation_deg,
                    self.max_interval_s,
                    self.min_interval_s,
                    self.timeout_distance_m,
                    self.timeout_rotation_deg,
                )
            )
            or self.min_interval_s > self.max_interval_s
            or self.timeout_distance_m > self.distance_m
            or self.timeout_rotation_deg > self.rotation_deg
        ):
            raise ValueError("invalid keyframe thresholds")

    def decide(self, stamp_ns, pose, previous, *, body_pose=None):
        result = {
            "reasons": [],
            "distance_m": None,
            "rotation_deg": None,
            "interval_s": None,
            "pose_source": None,
            "status": "first",
        }
        if previous is None:
            result["reasons"] = ["first"]
            return result
        elapsed = (stamp_ns - previous["source_stamp_ns"]) / 1e9
        result["interval_s"] = elapsed
        if elapsed < self.min_interval_s:
            result["status"] = "time_rollback" if elapsed < 0 else "min_interval"
            return result
        before = previous.get("camera_pose")
        # Always compare the same kind of pose. A body fallback is sampling
        # evidence only; it must never populate camera_pose or spatial queries.
        if pose is not None and before is not None and pose["frame"] == before["frame"]:
            result["pose_source"] = "camera_pose"
        else:
            pose, before = body_pose, previous.get("body_pose")
            if (
                pose is not None
                and before is not None
                and pose["frame"] == before["frame"]
            ):
                result["pose_source"] = "body_pose"
        if result["pose_source"] is not None:
            result["distance_m"] = math.dist(
                [pose[k] for k in ("x", "y", "z")], [before[k] for k in ("x", "y", "z")]
            )
            result["rotation_deg"] = rotation_deg(
                [pose[k] for k in ("qx", "qy", "qz", "qw")],
                [before[k] for k in ("qx", "qy", "qz", "qw")],
            )
            if result["distance_m"] + 1e-9 >= self.distance_m:
                result["reasons"].append("distance")
            if result["rotation_deg"] + 1e-9 >= self.rotation_deg:
                result["reasons"].append("rotation")
            moved = (
                result["distance_m"] + 1e-9 >= self.timeout_distance_m
                or result["rotation_deg"] + 1e-9 >= self.timeout_rotation_deg
            )
            # A timeout supplements slow motion, rather than periodically
            # recording a parked robot or millimetre-scale SLAM jitter.
            if elapsed >= self.max_interval_s and moved:
                result["reasons"].append("timeout")
            result["status"] = (
                "selected"
                if result["reasons"]
                else "waiting_for_threshold"
                if moved
                else "stationary"
            )
        else:
            # Missing pose cannot establish that a robot is stationary. Retain
            # explicit time-only coverage for offline/no-SLAM operation.
            result["status"] = "pose_unavailable"
            if elapsed >= self.max_interval_s:
                result["reasons"].append("timeout")
        return result
