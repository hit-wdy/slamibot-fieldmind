"""The search service. MCP (and later HTTP) are thin wrappers over this.

Spec section 7 ordering is the important part: the camera/time/label/pose filter
runs in SQLite *first*, and only the surviving rows have their vectors loaded and
ranked. A global top-50 that is then filtered would silently drop the genuinely
highest-scoring frames inside the requested range.

Two things this deliberately does not claim:

* `similarity` is a cosine, not a probability. There is no calibrated threshold,
  so `threshold_calibrated` is always false and `min_similarity` defaults to None.
* A group is a set of candidate frames close in time on one camera. It is not an
  event boundary: the frames between two candidates were never examined.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from .encoder import AlignedEncoder
from .detection_types import COCO_CLASSES
from .pose import Pose
from .store import RecordStore, StoredFrame

DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 50
MAX_CANDIDATE_K = 2000
DEFAULT_FIND_LIMIT = 50
MAX_FIND_LIMIT = 1000
DEFAULT_GROUP_WINDOW_S = 2.0

STATUS_OK = "ok"
STATUS_NO_FRAMES = "no_frames_in_range"
STATUS_NO_MATCH = "no_match_above_threshold"


def verification_target(query: str, target_label: str | None) -> str:
    """Resolve an explicit target or a simple class-name query, without guessing attributes.

    For descriptive queries the MCP caller supplies the intended object class.
    A phrase such as 'fire truck' must not silently become a verified 'truck'.
    """
    target = " ".join((target_label if target_label is not None else query).lower().split()).strip(" .!?")
    if not target:
        raise ValueError("target_label must not be empty")
    if target_label is None:
        target = re.sub(r"^(?:a |an )?(?:photo|image|picture) of ", "", target)
        target = re.sub(r"^(?:a|an|the) ", "", target)
    return target


def verify_detection(frame: StoredFrame, target: str) -> dict:
    """Annotate a RADIO candidate using the same frame's stored YOLO detections."""
    if target not in COCO_CLASSES:
        return {
            "status": "not_supported",
            "target_label": target,
            "detections": [],
            "note": "No exact YOLO class for this target; return the best RADIO candidates as ranked.",
        }
    detections = [d.to_dict() for d in frame.detections if d.label == target]
    return {
        "status": "supported_by_yolo" if detections else "not_detected",
        "target_label": target,
        "detections": detections,
        "note": (
            "YOLO supports this object class in this frame, not the query's attributes or relations."
            if detections else
            "YOLO did not report this class in this frame; this is not proof of absence. Keep the RADIO candidate."
        ),
    }


def parse_time(value: str | dt.datetime | None) -> dt.datetime | None:
    """Accept ISO 8601 with an offset, or UTC if none is given."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{value!r} is not an ISO 8601 timestamp") from error
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Ranked:
    frame: StoredFrame
    similarity: float


class QueryService:
    """Reads a record collection. Needs no encoder until a text query arrives.

    No SQLite connection is held between calls. A server runs tool handlers on
    whichever worker thread is free, and a sqlite3 connection may only be used by
    the thread that created it, so a cached connection works until the second
    tool call lands on a different thread and then fails.
    """

    def __init__(
        self,
        root: Path | str,
        collection_id: str,
        encoder_factory: Callable[[], AlignedEncoder] | None = None,
    ) -> None:
        self.root = Path(root)
        self.collection_id = collection_id
        self._encoder_factory = encoder_factory
        self._encoder: AlignedEncoder | None = None
        self._encoder_lock = threading.Lock()

    # -- plumbing ---------------------------------------------------------------

    @contextmanager
    def store(self) -> Iterator[RecordStore]:
        """A short-lived connection, opened and closed inside the calling thread."""
        opened = RecordStore.open_existing(self.root, self.collection_id)
        try:
            yield opened
        finally:
            opened.close()

    def _loaded_encoder(self) -> AlignedEncoder:
        """Caller must hold `_encoder_lock`."""
        if self._encoder is None:
            if self._encoder_factory is None:
                raise RuntimeError("no encoder configured; this collection cannot be searched by text")
            self._encoder = self._encoder_factory()
        return self._encoder

    def encoder(self) -> AlignedEncoder:
        """Loaded on first text query: ~20 s and ~1.2 GB that most tools never need."""
        with self._encoder_lock:
            return self._loaded_encoder()

    def encode_text(self, text: str) -> np.ndarray:
        """Serialised: one GPU at a time, as spec section 3 requires."""
        with self._encoder_lock:
            return self._loaded_encoder().encode_texts([text])[0]

    def close(self) -> None:
        with self._encoder_lock:
            self._encoder = None

    def __enter__(self) -> "QueryService":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- helpers ----------------------------------------------------------------

    def _resolve_range(
        self,
        start: dt.datetime | None,
        end: dt.datetime | None,
        start_time: str | dt.datetime | None,
        end_time: str | dt.datetime | None,
    ) -> tuple[dt.datetime, dt.datetime]:
        end = parse_time(end_time) or end
        start = parse_time(start_time) or start
        if end is None:
            end = dt.datetime.now(dt.timezone.utc)
        if start is None:
            start = end - dt.timedelta(hours=DEFAULT_WINDOW_HOURS)
        if start >= end:
            raise ValueError(f"start_time {iso(start)} is not before end_time {iso(end)}")
        return start, end

    def _resolve_near(
        self,
        store: RecordStore,
        near_pose: Pose | None,
        near_frame_id: str | None,
        radius_m: float | None,
    ) -> Pose | None:
        """Accept a reference frame id as well as raw coordinates.

        A language model should not have to emit a quaternion to ask "what was
        near this frame", so `near_frame_id` resolves through the stored pose.
        """
        if near_frame_id is not None:
            if near_pose is not None:
                raise ValueError("give either near_frame_id or near_pose, not both")
            frame = store.get_frame(near_frame_id)
            if frame is None:
                raise ValueError(f"reference frame {near_frame_id!r} is not in this collection")
            if frame.pose.source == "unavailable":
                raise ValueError("reference frame has no camera pose")
            near_pose = frame.pose
        if near_pose is not None and radius_m is None:
            raise ValueError("a pose filter needs radius_m, e.g. 2.0 for two metres")
        return near_pose

    def _empty_result_note(
        self,
        store: RecordStore,
        start: dt.datetime,
        end: dt.datetime,
        camera_ids: Sequence[str] | None,
        near_pose: Pose | None,
        radius_m: float | None,
        labels: Sequence[str] | None,
    ) -> dict:
        """Explain an empty result. Two very different causes, two different notes.

        * the window itself holds nothing -- the caller should widen the times;
        * the window holds frames but the filters excluded them all -- the frames
          exist and a missing label is not proof the object was absent.

        Conflating these would mislead in one direction or the other, and a harness
        reading "no frames" would draw exactly the wrong conclusion.
        """
        first, last = store.time_range()
        if first is None:
            return {
                "collection_is_empty": True,
                "empty_because": "no_records",
                "note": "this collection holds no records at all; nothing has been recorded yet",
            }

        base: dict[str, Any] = {
            "collection_is_empty": False,
            "collection_total_frames": store.count(),
            "collection_time_range": {"earliest": iso(first), "latest": iso(last)},
        }
        in_window = self._filtered(store, camera_ids, start, end, None, near_pose, radius_m)
        if not in_window:
            base.update(
                {
                    "empty_because": "time_window",
                    "hours_since_latest_record": round(
                        (dt.datetime.now(dt.timezone.utc) - last).total_seconds() / 3600, 2
                    ),
                    "note": (
                        f"the requested window {iso(start)} .. {iso(end)} holds no records, but the "
                        f"collection holds {store.count()} frames from {iso(first)} to {iso(last)}. "
                        "Widen start_time and end_time. An empty window is not evidence that nothing was seen."
                    ),
                }
            )
        else:
            base.update(
                {
                    "empty_because": "filters",
                    "frames_in_window_before_filters": len(in_window),
                    "note": (
                        f"the window holds {len(in_window)} frames but none matched the filters "
                        f"(labels={list(labels) if labels else None}, radius_m={radius_m}). The frames "
                        "exist and the filter is what excluded them; a missing label is not proof the "
                        "object was absent."
                    ),
                }
            )
        return base

    def _filtered(
        self,
        store: RecordStore,
        camera_ids: Sequence[str] | None,
        start: dt.datetime,
        end: dt.datetime,
        labels: Sequence[str] | None,
        near_pose: Pose | None,
        radius_m: float | None,
        limit: int | None = None,
    ) -> list[StoredFrame]:
        """SQL-side filtering first, then any pose radius in Python.

        Only single-camera pose filtering is possible without an index on the
        position; `Pose.distance_to` refuses to compare across frames anyway, so a
        multi-camera request drops the radius rather than pretending.
        """
        cameras = list(camera_ids) if camera_ids else [None]
        frames: list[StoredFrame] = []
        for camera in cameras:
            frames.extend(
                store.iter_frames(camera_id=camera, start=start, end=end, labels=labels, limit=limit)
            )
        frames.sort(key=lambda f: f.captured_at)
        if near_pose is not None:
            if radius_m is None:
                raise ValueError("a pose filter needs radius_m")
            frames = [
                f
                for f in frames
                if f.pose.source != "unavailable" and f.pose.frame == near_pose.frame and f.pose.distance_to(near_pose) <= radius_m
            ]
        return frames

    @staticmethod
    def _frame_payload(
        frame: StoredFrame,
        similarity: float | None,
        location: str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "frame_id": frame.frame_id,
            "camera_id": frame.camera_id,
            "location": location,
            "captured_at": iso(frame.captured_at),
            "device_timestamp_ns": frame.device_timestamp_ns,
            "pose": frame.pose.to_dict() if frame.pose.source != "unavailable" else None,
            "detections": [d.to_dict() for d in frame.detections],
            "label_counts": frame.label_counts,
        }
        if frame.pose.source == "unavailable":
            payload["pose_note"] = "No measured camera pose is available."
        elif frame.pose.source != "slam":
            payload["pose_is_a_stand_in"] = (
                "this pose comes from a simulated trajectory, not the SLAM system; it does not "
                "locate anything in the real world"
            )
        if similarity is not None:
            payload["similarity"] = round(float(similarity), 4)
        return payload

    @staticmethod
    def _group(ranked: list[Ranked], window_s: float) -> list[tuple[Ranked, list[Ranked]]]:
        """Merge candidates on one camera that are close in time.

        Greedy in similarity order: the strongest candidate founds a group, later
        ones join the first group on their camera within `window_s`.
        """
        groups: list[tuple[Ranked, list[Ranked]]] = []
        for item in ranked:
            for representative, members in groups:
                if representative.frame.camera_id != item.frame.camera_id:
                    continue
                gap = abs((item.frame.captured_at - representative.frame.captured_at).total_seconds())
                if gap <= window_s:
                    members.append(item)
                    break
            else:
                groups.append((item, [item]))
        return groups

    # -- tools ------------------------------------------------------------------

    def search(
        self,
        query: str,
        camera_ids: Sequence[str] | None = None,
        start_time: str | dt.datetime | None = None,
        end_time: str | dt.datetime | None = None,
        labels: Sequence[str] | None = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        min_similarity: float | None = None,
        group_results: bool = True,
        group_window_s: float = DEFAULT_GROUP_WINDOW_S,
        near_pose: Pose | None = None,
        near_frame_id: str | None = None,
        radius_m: float | None = None,
        target_label: str | None = None,
    ) -> dict:
        """Semantic search, followed by YOLO evidence on each returned frame.

        target_label is a verification target, not a pre-filter or ranking signal.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        target = verification_target(query, target_label)
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if candidate_k < top_k:
            raise ValueError(f"candidate_k ({candidate_k}) must be at least top_k ({top_k})")
        candidate_k = min(candidate_k, MAX_CANDIDATE_K)

        start, end = self._resolve_range(None, None, start_time, end_time)
        with self.store() as store:
            near_pose = self._resolve_near(store, near_pose, near_frame_id, radius_m)
            frames = self._filtered(store, camera_ids, start, end, labels, near_pose, radius_m)
            locations = store.camera_locations()
            if frames:
                matrix = store.load_vectors(frames)
                empty = None
            else:
                matrix = None
                empty = self._empty_result_note(
                    store, start, end, camera_ids, near_pose, radius_m, labels
                )

        response: dict[str, Any] = {
            "query": query,
            "collection_id": self.collection_id,
            "start_time": iso(start),
            "end_time": iso(end),
            "threshold_calibrated": False,
            "calibration_note": (
                "similarity is a cosine in the RADIO/CLIP space, not a probability. "
                "No threshold has been calibrated on labelled data, so these are ranked "
                "candidates. Do not report them as detection or identification."
            ),
            "indexed_frame_count_in_range": len(frames),
            "pending_frame_count_in_range": 0,
            "retrieval_strategy": "radio_then_yolo" if target in COCO_CLASSES else "radio_only",
            "verification_target": target,
            "verification_note": (
                "Candidates remain in RADIO similarity order. YOLO evidence is checked after retrieval; "
                "a missing detection does not remove a candidate. For descriptive queries, supply "
                "target_label as the intended COCO object class, or its unsupported object name."
            ),
            "grouped": bool(group_results),
            "group_window_s": group_window_s if group_results else None,
            "matches": [],
        }
        if not frames:
            response["status"] = STATUS_NO_FRAMES
            response.update(empty or {})
            return response

        started_at = frames[0].captured_at
        end_cap = frames[-1]
        response["range_covered_by_index"] = {
            "earliest": iso(started_at),
            "latest": iso(end_cap.captured_at),
        }

        query_vector = self.encode_text(query)
        scores = matrix @ query_vector

        order = np.argsort(-scores)[:candidate_k]
        ranked = [Ranked(frames[int(i)], float(scores[int(i)])) for i in order]

        if min_similarity is not None:
            above = [item for item in ranked if item.similarity >= min_similarity]
            if not above:
                response["status"] = STATUS_NO_MATCH
                response["min_similarity"] = min_similarity
                response["note"] = (
                    "frames exist in the range but none reached min_similarity; the range was searched, "
                    "nothing cleared the (uncalibrated) threshold"
                )
                return response
            ranked = above

        if group_results:
            groups = self._group(ranked, group_window_s)[:top_k]
            for index, (representative, members) in enumerate(groups, start=1):
                payload = self._frame_payload(
                    representative.frame, representative.similarity, locations.get(representative.frame.camera_id)
                )
                payload["group_id"] = f"group-{index:02d}"
                payload["matched_frame_count"] = len(members)
                payload["group_time_span_s"] = round(
                    max(
                        abs((m.frame.captured_at - representative.frame.captured_at).total_seconds())
                        for m in members
                    ),
                    2,
                )
                response["matches"].append(payload)
            response["grouping_note"] = (
                "each match stands for candidates close in time on one camera; the frames between "
                "candidates were not examined and the span is not an event boundary"
            )
        else:
            for item in ranked[:top_k]:
                response["matches"].append(
                    self._frame_payload(item.frame, item.similarity, locations.get(item.frame.camera_id))
                )

        response["status"] = STATUS_OK
        response["min_similarity"] = min_similarity
        by_id = {frame.frame_id: frame for frame in frames}
        for match in response["matches"]:
            match["yolo_verification"] = verify_detection(by_id[match["frame_id"]], target)
        return response

    def find_frames(
        self,
        labels: Sequence[str] | None = None,
        camera_ids: Sequence[str] | None = None,
        start_time: str | dt.datetime | None = None,
        end_time: str | dt.datetime | None = None,
        near_pose: Pose | None = None,
        near_frame_id: str | None = None,
        radius_m: float | None = None,
        limit: int = DEFAULT_FIND_LIMIT,
    ) -> dict:
        """Metadata-only lookup: which frames hold these labels, where, when.

        No encoder is involved. `labels` are YOLO class names and are only as
        reliable as the detector: on a static scene the detected set fluctuates
        frame to frame, so a label is evidence, not a fact.
        """
        if not 1 <= limit <= MAX_FIND_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_FIND_LIMIT}")
        start, end = self._resolve_range(None, None, start_time, end_time)
        with self.store() as store:
            near_pose = self._resolve_near(store, near_pose, near_frame_id, radius_m)
            frames = self._filtered(store, camera_ids, start, end, labels, near_pose, radius_m)
            locations = store.camera_locations()
            empty = (
                self._empty_result_note(store, start, end, camera_ids, near_pose, radius_m, labels)
                if not frames
                else None
            )
        total = len(frames)
        shown = frames[:limit]
        return {
            "collection_id": self.collection_id,
            "status": STATUS_OK if total else STATUS_NO_FRAMES,
            **(empty or {}),
            "labels": list(labels) if labels else None,
            "start_time": iso(start),
            "end_time": iso(end),
            "matched_frame_count": total,
            "returned": len(shown),
            "truncated": total > len(shown),
            "caveat": (
                "labels come from a YOLO detector and carry false positives and frame-to-frame "
                "flicker; a frame without a label is not proof the object was absent"
            ),
            "frames": [self._frame_payload(f, None, locations.get(f.camera_id)) for f in shown],
        }

    def get_frame(self, frame_id: str) -> dict:
        """One record in full. No image exists: only the embedding, detections and pose."""
        with self.store() as store:
            frame = store.get_frame(frame_id)
            dimension = store.embedding_dim
            locations = store.camera_locations()
        if frame is None:
            return {"status": "not_found", "frame_id": frame_id, "collection_id": self.collection_id}
        payload = self._frame_payload(frame, None, locations.get(frame.camera_id))
        payload["status"] = STATUS_OK
        payload["vector"] = {
            "partition": frame.vector_partition,
            "row": frame.vector_row,
            "dim": dimension,
            "dtype_on_disk": "float16",
        }
        payload["note"] = "no image is stored for this record; only the embedding, detections and pose"
        return payload

    def compare_frames(self, frame_ids: Sequence[str]) -> dict:
        """Compare records: time gaps, pose separation, and which labels differ."""
        if len(frame_ids) < 2:
            raise ValueError("compare_frames needs at least two frame ids")
        frames = []
        missing = []
        with self.store() as store:
            for frame_id in frame_ids:
                frame = store.get_frame(frame_id)
                (frames.append(frame) if frame else missing.append(frame_id))
        if missing:
            return {"status": "not_found", "missing_frame_ids": missing}

        frames.sort(key=lambda f: f.captured_at)
        reference = frames[0]
        rows = []
        for frame in frames:
            row = {
                "frame_id": frame.frame_id,
                "captured_at": iso(frame.captured_at),
                "seconds_from_first": round((frame.captured_at - reference.captured_at).total_seconds(), 3),
                "camera_id": frame.camera_id,
                "pose": frame.pose.to_dict() if frame.pose.source != "unavailable" else None,
                "label_counts": frame.label_counts,
            }
            if frame.pose.source != "unavailable" and reference.pose.source != "unavailable" and frame.pose.frame == reference.pose.frame:
                row["distance_from_first_m"] = round(reference.pose.distance_to(frame.pose), 3)
                row["heading_change_from_first_deg"] = round(reference.pose.yaw_change_to(frame.pose), 1)
            else:
                row["distance_from_first_m"] = None
                row["heading_change_from_first_deg"] = None
            rows.append(row)

        labels_seen = {f.frame_id: set(f.label_counts) for f in frames}
        common = set.intersection(*labels_seen.values()) if labels_seen else set()
        return {
            "status": STATUS_OK,
            "collection_id": self.collection_id,
            "pose_source": reference.pose.source,
            "pose_source_note": (
                "simulated poses are a stand-in for the SLAM system; distances and headings "
                "computed from them describe the simulation, not the camera"
                if reference.pose.source != "slam"
                else "measured"
            ),
            "frames": rows,
            "labels_in_every_frame": sorted(common),
            "labels_not_in_every_frame": sorted(set.union(*labels_seen.values()) - common) if labels_seen else [],
        }

    def status(self) -> dict:
        """Coverage and health. 'The process is up' is not the same as 'the index is current'."""
        with self.store() as store:
            frames = list(store.iter_frames())
            integrity = store.verify()
            fingerprint = store.model_fingerprint
            dimension = store.embedding_dim
            labels = store.stats()["detection_labels"]
            locations = store.camera_locations()
        by_camera: dict[str, dict] = {}
        for frame in frames:
            entry = by_camera.setdefault(
                frame.camera_id,
                {"frames": 0, "first": None, "last": None, "location": locations.get(frame.camera_id)},
            )
            entry["frames"] += 1
            stamp = iso(frame.captured_at)
            entry["first"] = entry["first"] or stamp
            entry["last"] = stamp
        latest = frames[-1].captured_at if frames else None
        return {
            "status": STATUS_OK,
            "collection_id": self.collection_id,
            "model_fingerprint": fingerprint,
            "embedding_dim": dimension,
            "indexed_frames": len(frames),
            "cameras": by_camera,
            "latest_record_at": iso(latest) if latest else None,
            "seconds_since_latest_record": round(
                (dt.datetime.now(dt.timezone.utc) - latest).total_seconds(), 1
            )
            if latest
            else None,
            "detection_labels": labels,
            "pose_source": frames[-1].pose.source if frames else None,
            "stores_images": False,
            "vector_partitions_ok": integrity["ok"],
            "integrity_problems": integrity["orphan_or_missing"],
            "threshold_calibrated": False,
        }
