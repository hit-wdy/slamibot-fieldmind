"""Camera pose at capture time.

In the real deployment a SLAM system supplies this. Until then
`SimulatedPoseSource` stands in, and every record says so via `Pose.source`, so
a simulated pose can never be mistaken for a measured one downstream.

The pose is stored the way a SLAM front end would report it: a translation and a
rotation quaternion in a named coordinate frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

SIMULATED = "simulated"
SLAM = "slam"


@dataclass(frozen=True)
class Pose:
    """Translation and orientation of the camera in `frame`."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    frame: str = "map"
    source: str = SIMULATED

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "qx": round(self.qx, 6),
            "qy": round(self.qy, 6),
            "qz": round(self.qz, 6),
            "qw": round(self.qw, 6),
            "frame": self.frame,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def yaw_degrees(self) -> float:
        """Heading in degrees, from the quaternion, assuming near-level motion."""
        return math.degrees(math.atan2(2 * (self.qw * self.qz + self.qx * self.qy),
                                       1 - 2 * (self.qy**2 + self.qz**2)))

    def distance_to(self, other: "Pose") -> float:
        """Straight-line distance in metres. Meaningless across different frames."""
        if self.frame != other.frame:
            raise ValueError(f"cannot compare poses in different frames: {self.frame!r} vs {other.frame!r}")
        return math.dist(self.position, other.position)

    def yaw_change_to(self, other: "Pose") -> float:
        """Heading change in degrees, wrapped to (-180, 180]."""
        delta = (other.yaw_degrees() - self.yaw_degrees() + 180.0) % 360.0 - 180.0
        return delta


class SimulatedPoseSource:
    """A stand-in trajectory: a slow circle with a small vertical bob.

    Deterministic in elapsed time, so two runs with the same seed and the same
    timestamps produce the same poses -- which is what makes the stored records
    comparable between runs. This is scaffolding for the SLAM integration, not a
    model of any real camera motion.
    """

    def __init__(
        self,
        seed: int = 0,
        radius_m: float = 1.5,
        height_m: float = 1.2,
        period_s: float = 60.0,
        start_time: float = 0.0,
        frame: str = "map",
    ) -> None:
        self.seed = seed
        self.radius_m = radius_m
        self.height_m = height_m
        self.period_s = period_s
        self.start_time = start_time
        self.frame = frame

    def at(self, elapsed_s: float) -> Pose:
        if self.period_s <= 0:
            raise ValueError("period_s must be positive")
        t = elapsed_s - self.start_time
        phase = 2 * math.pi * t / self.period_s

        x = self.radius_m * math.cos(phase) + 0.05 * self.seed
        y = self.radius_m * math.sin(phase)
        z = self.height_m + 0.05 * math.sin(2 * phase)

        # Face along the direction of travel, with a slow scan on top.
        yaw = phase + math.pi / 2 + 0.20 * math.sin(3 * phase) + 0.01 * self.seed
        return Pose(
            x=x,
            y=y,
            z=z,
            qx=0.0,
            qy=0.0,
            qz=math.sin(yaw / 2),
            qw=math.cos(yaw / 2),
            frame=self.frame,
            source=SIMULATED,
        )

    def describe(self) -> dict:
        return {
            "source": SIMULATED,
            "note": (
                "stand-in for the SLAM pose; a real deployment must replace this "
                "with the measured camera pose for the same timestamp"
            ),
            "frame": self.frame,
            "seed": self.seed,
            "radius_m": self.radius_m,
            "height_m": self.height_m,
            "period_s": self.period_s,
        }
