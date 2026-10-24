"""ROS-free demo state and evidence helpers; also copied to Python 3.8 guests."""

import math
import threading
import time


def simulated_pose(seconds):
    """Same synthetic trajectory as semantic_perception/simulated_pose_node.py."""
    cycle, phase = divmod(max(0.0, seconds), 40.0)
    x = cycle * 10 + max(0, min(phase - 10, 10)) * .5
    x += max(0, phase - 30) * .5
    yaw = math.radians(cycle * 200 + max(0, min(phase - 20, 10)) * 20)
    return dict(x=x, y=0., z=0., qx=0., qy=0., qz=math.sin(yaw / 2),
                qw=math.cos(yaw / 2), frame="simulated_map", source="simulated")


class ReplayControl:
    """Wall time while armed/paused is never charged to the media timeline."""
    def __init__(self, speed=1., clock=time.monotonic):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        self.speed, self.clock = speed, clock
        self.lock = threading.RLock()
        self.state, self.started = "initializing", False
        self.position, self.updated = 0., clock()

    def _advance(self):
        now = self.clock()
        if now < self.updated:
            raise ValueError("monotonic clock moved backwards")
        if self.state == "playing":
            self.position += (now - self.updated) * self.speed
        self.updated = now

    def ready(self):
        with self.lock:
            if self.state != "initializing":
                raise ValueError("cannot arm from " + self.state)
            self._advance()
            self.state = "paused"

    def command(self, action):
        with self.lock:
            self._advance()
            if action == "start-navigation" and self.state == "paused" and not self.started:
                self.started, self.state = True, "playing"
            elif action == "pause" and self.state == "playing":
                self.state = "paused"
            elif action == "resume" and self.state == "paused" and self.started:
                self.state = "playing"
            elif action == "stop" and self.state != "failed":
                self.state = "stopped"
            else:
                raise ValueError("invalid action %s in %s" % (action, self.state))
            return self.snapshot()

    def finish(self, state="complete"):
        with self.lock:
            self._advance()
            if state not in ("complete", "failed"):
                raise ValueError("invalid terminal state")
            if self.state != "stopped":
                self.state = state

    def drain(self):
        with self.lock:
            self._advance()
            if self.state not in ("playing", "paused"):
                raise ValueError("cannot drain from " + self.state)
            self.state = "indexing"

    def snapshot(self):
        with self.lock:
            self._advance()
            return dict(state=self.state, navigation_started=self.started,
                        playback_elapsed_s=round(self.position, 3), speed=self.speed)


def navigation_points(coordinate_id, token, duration):
    return [dict(place_id="demo-%s-position-%d" % (token, i), name="位置" + ("一", "二", "三", "四")[i-1],
                 coordinate_id=coordinate_id, radius_m=3., offset_s=offset,
                 **{k: v for k, v in simulated_pose(offset).items() if k in ("x", "y", "z", "frame")})
            for i, offset in enumerate((30., 90., 150., 210.), 1) if offset <= duration]


def relative_place(frame, places):
    pose = frame.get("camera_pose")
    if not pose:
        return None
    valid = [p for p in places if p.get("coordinate_id") == frame.get("coordinate_id")
             and p.get("frame") == pose.get("frame")]
    if not valid:
        return None
    point = min(valid, key=lambda p: sum((pose[k] - p[k]) ** 2 for k in ("x", "y", "z")))
    delta = {k: round(pose[k] - point[k], 3) for k in ("x", "y", "z")}
    return dict(place_id=point["place_id"], name=point["name"],
                camera_distance_m=round(math.sqrt(sum(v*v for v in delta.values())), 3),
                camera_offset_map_m=delta, pose_source=pose.get("source"),
                note="Observing-camera position, not object location; demo coordinates are synthetic.")


def class_count(frame, label):
    if frame.get("detection_status") != "available":
        return None
    return sum(d["label"] == label for d in frame.get("detections", []))


def choose_candidates(matches, label, exact_count=None):
    """Preserve RADIO ordering; count/attributes are explicit agent-side evidence."""
    return [dict(row, radio_rank=rank) for rank, row in enumerate(matches, 1)
            if (class_count(row, label) == exact_count if exact_count is not None
                else (class_count(row, label) or 0) > 0)]
