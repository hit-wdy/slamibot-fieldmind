"""YOLO scene inventory: a live view or the latest observation at a registered place."""

from __future__ import annotations

import datetime as dt
import math
import threading
from pathlib import Path
from typing import Any, Callable

from .detection_types import summarise
from .store import RecordStore


class SceneService:
    def __init__(self, root: Path | str, collection_id: str, source_factory: Callable[..., Any] | None = None):
        self.root = Path(root)
        self.collection_id = collection_id
        self._source_factory = source_factory
        self._live_lock = threading.Lock()

    def describe(
        self,
        mode: str = "current",
        location_label: str | None = None,
        camera_id: str | None = None,
        device_id: str | None = None,
        max_age_s: float = 5.0,
    ) -> dict:
        if mode not in {"current", "location"}:
            raise ValueError("mode must be current or location")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if location_label is not None:
            location_label = location_label.strip()
            if not location_label:
                raise ValueError("location_label must not be empty")
        if mode == "current" and location_label is not None:
            raise ValueError("current view cannot claim a named location; use mode=location for stored observations")
        if mode == "location" and not (location_label or camera_id):
            raise ValueError("location mode requires location_label or a navigation-resolved camera_id")
        if mode == "location" and device_id is not None:
            raise ValueError("device_id is only used for current view")

        base = {
            "mode": mode,
            "location_label": location_label,
            "max_age_s": max_age_s,
            "radio_used": False,
            "caveat": (
                "These are YOLO detections within its supported classes and central crop, not a complete "
                "scene description. Counts are per frame, not unique objects over time or cameras. "
                "No detections is not proof that nothing is present."
            ),
        }
        if mode == "current":
            return {**base, **self._current(camera_id or "camera_01", device_id, max_age_s)}
        return {**base, **self._location(location_label, camera_id, max_age_s)}

    @staticmethod
    def _freshness(captured_at: dt.datetime, max_age_s: float) -> dict:
        age = max(0.0, (dt.datetime.now(dt.timezone.utc) - captured_at).total_seconds())
        return {"age_s": round(age, 3), "stale": age > max_age_s}

    def _current(self, camera_id: str, device_id: str | None, max_age_s: float) -> dict:
        if not self._live_lock.acquire(blocking=False):
            return {"status": "camera_busy", "source": "live_camera", "views": []}
        source = None
        try:
            if self._source_factory is None:
                from .capture import CameraSource
                from .detection_types import YoloV6Detector

                source = CameraSource(camera_id=camera_id, device_id=device_id, fps=10.0,
                                      detector=YoloV6Detector(), queue_size=2)
            else:
                source = self._source_factory(camera_id=camera_id, device_id=device_id)
            source.open()
            source.flush()
            for _ in range(3):
                frame = source.grab()
                if frame.detections_available:
                    break
            freshness = self._freshness(frame.captured_at, max_age_s)
            view = {
                "camera_id": frame.camera_id,
                "location": None,
                "captured_at": frame.captured_at.isoformat(timespec="milliseconds"),
                "timestamp_source": frame.timestamp_source,
                "source_sequence": frame.source_sequence,
                "device_timestamp_ns": frame.device_timestamp_ns,
                "detection_status": "available" if frame.detections_available else "unavailable",
                "label_counts": summarise(frame.detections) if frame.detections_available else None,
                "detections": [d.to_dict() for d in frame.detections] if frame.detections_available else None,
                "exposure": source.settle_info,
                **freshness,
            }
            status = "detection_unavailable" if not frame.detections_available else (
                "stale" if freshness["stale"] else "ok"
            )
            return {"status": status, "source": "live_camera", "views": [view], "stored": False}
        except (RuntimeError, TimeoutError, OSError) as error:
            return {
                "status": "camera_unavailable", "source": "live_camera", "views": [],
                "error": str(error),
                "note": "Could not obtain a current view. Check device connection or an active recorder; no old record was substituted.",
            }
        finally:
            try:
                if source is not None:
                    source.close()
            finally:
                self._live_lock.release()

    def _location(self, location_label: str | None, camera_id: str | None, max_age_s: float) -> dict:
        with RecordStore.open_existing(self.root, self.collection_id) as store:
            locations = store.camera_locations()
            selected = [cid for cid, location in locations.items()
                        if (camera_id is None or cid == camera_id)
                        and (location_label is None or location == location_label)]
            views = []
            missing = []
            for cid in sorted(selected):
                row = store.connection.execute(
                    "SELECT frame_id FROM frames WHERE collection_id=? AND camera_id=? "
                    "ORDER BY captured_at DESC, source_sequence DESC LIMIT 1",
                    (self.collection_id, cid),
                ).fetchone()
                if row is None:
                    missing.append(cid)
                    continue
                frame = store.get_frame(row["frame_id"])
                views.append({
                    "frame_id": frame.frame_id,
                    "camera_id": frame.camera_id,
                    "location": locations[cid],
                    "captured_at": frame.captured_at.isoformat(timespec="milliseconds"),
                    "label_counts": frame.label_counts,
                    "detections": [d.to_dict() for d in frame.detections],
                    "detection_status": "recorded",
                    **self._freshness(frame.captured_at, max_age_s),
                })
        if not selected:
            status = "location_not_found"
        elif not views:
            status = "no_observations"
        elif any(v["stale"] for v in views):
            status = "stale"
        elif missing:
            status = "partial"
        else:
            status = "ok"
        return {
            "status": status, "source": "latest_record", "views": views,
            "cameras_without_observations": missing,
            "note": (
                "Latest stored observation per registered camera, not a live remote view. "
                "Place matching uses the camera's registered label, not a navigation map or historical "
                "per-frame location. Stored empty detections may also mean a missing detector result."
            ),
        }
