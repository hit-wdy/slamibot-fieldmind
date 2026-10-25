"""Queries over complete and pending keyframes, with explicit coordinate scope."""

from __future__ import annotations

import datetime as dt
import io
import math

import numpy as np
from PIL import Image

from .annotation import draw_detections

from .detection_types import COCO_CLASSES
from .keyframes import rotation_deg
from .query import parse_time, verification_target
from .semantic_store import connect, payload
from .store import RecordStore
from .visual_evidence import attach_visual_evidence


def render_jpeg(source, max_edge=1280, detections=None):
    with Image.open(source) as original:
        image = original.convert("RGB")
        image.thumbnail((max_edge, max_edge))
        if detections is not None:
            draw_detections(image, detections)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue()


def freshness(stamp, max_age_s):
    if not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_age_s must be finite and positive")
    age = (dt.datetime.now(dt.timezone.utc) - parse_time(stamp)).total_seconds()
    return dict(
        captured_at=stamp,
        age_s=round(max(0, age), 3),
        stale=age > max_age_s or age < -1,
        clock_in_future=age < -1,
    )


class BackendQuery:
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.config.root
        self.collection_id = runtime.config.collection_id

    def list_places(self):
        with connect(self.root) as db:
            return {
                "places": [
                    dict(r)
                    for r in db.execute("SELECT * FROM places ORDER BY name,place_id")
                ]
            }

    def list_sessions(self):
        with connect(self.root) as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT s.*,COUNT(o.frame_id) AS frames,MIN(o.captured_at) AS first_observation,MAX(o.captured_at) AS last_observation FROM sessions s LEFT JOIN observations o ON o.session_id=s.session_id AND o.collection_id=? GROUP BY s.session_id ORDER BY s.started_at DESC",
                    (self.collection_id,),
                )
            ]
        return {"sessions": rows}

    def _rows(
        self,
        camera_ids=None,
        start_time=None,
        end_time=None,
        labels=None,
        task_id=None,
        session_id=None,
        coordinate_id=None,
        place_id=None,
        near_frame_id=None,
        radius_m=None,
        x=None,
        y=None,
        z=None,
        frame=None,
        ready_only=False,
    ):
        end = (parse_time(end_time) or dt.datetime.now(dt.timezone.utc)).astimezone(
            dt.timezone.utc
        )
        start = (parse_time(start_time) or end - dt.timedelta(hours=24)).astimezone(
            dt.timezone.utc
        )
        if start >= end:
            raise ValueError("start_time must precede end_time")
        clauses = ["collection_id=?", "captured_at>=?", "captured_at<?"]
        params = [
            self.collection_id,
            start.isoformat(timespec="milliseconds"),
            end.isoformat(timespec="milliseconds"),
        ]
        for key, value in (
            ("task_id", task_id),
            ("session_id", session_id),
            ("coordinate_id", coordinate_id),
        ):
            if value is not None:
                clauses.append(key + "=?")
                params.append(value)
        if camera_ids:
            clauses.append("camera_id IN (" + ",".join("?" for _ in camera_ids) + ")")
            params.extend(camera_ids)
        if ready_only:
            clauses.append("radio_status='ready'")
        if labels:
            clauses.append(
                "EXISTS(SELECT 1 FROM json_each(json_extract(metadata_json,'$.detections')) d WHERE json_extract(d.value,'$.label') IN ("
                + ",".join("?" for _ in labels)
                + "))"
            )
            params.extend(labels)
        spatial_sources = sum(
            [
                place_id is not None,
                near_frame_id is not None,
                any(v is not None for v in (x, y, z, frame)),
            ]
        )
        if spatial_sources > 1:
            raise ValueError("choose place_id, near_frame_id, or explicit coordinates")
        with connect(self.root) as db:
            spatial_coordinate = coordinate_id
            if place_id:
                place = db.execute(
                    "SELECT * FROM places WHERE place_id=?", (place_id,)
                ).fetchone()
                if not place:
                    raise ValueError("unknown place_id")
                x, y, z, frame, spatial_coordinate = (
                    place[k] for k in ("x", "y", "z", "frame", "coordinate_id")
                )
                radius_m = radius_m if radius_m is not None else place["radius_m"]
            elif near_frame_id:
                ref = self.get_frame(near_frame_id)
                p = ref.get("camera_pose")
                if p is None or ref.get("coordinate_id") is None:
                    raise ValueError(
                        "reference frame has no measured coordinate-scoped camera pose"
                    )
                x, y, z, frame = (p[k] for k in ("x", "y", "z", "frame"))
                spatial_coordinate = ref["coordinate_id"]
            if spatial_sources:
                if any(
                    v is None for v in (x, y, z, frame, spatial_coordinate, radius_m)
                ):
                    raise ValueError(
                        "spatial query needs x,y,z,frame,coordinate_id,radius_m"
                    )
                if (
                    not all(math.isfinite(float(v)) for v in (x, y, z, radius_m))
                    or radius_m <= 0
                ):
                    raise ValueError("invalid spatial coordinates or radius")
                clauses.extend(
                    [
                        "coordinate_id=?",
                        "json_extract(metadata_json,'$.camera_pose.frame')=?",
                    ]
                )
                params.extend([spatial_coordinate, frame])
                terms = []
                for key, value in zip(("x", "y", "z"), (x, y, z)):
                    expression = (
                        f"(json_extract(metadata_json,'$.camera_pose.{key}')-?)"
                    )
                    terms.append(expression + "*" + expression)
                    params.extend([float(value), float(value)])
                clauses.append("(" + "+".join(terms) + ")<=?")
                params.append(radius_m**2)
            elif radius_m is not None:
                raise ValueError("radius_m needs a spatial reference")
            rows = [
                payload(r)
                for r in db.execute(
                    "SELECT * FROM observations WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY captured_at,frame_id",
                    params,
                )
            ]
        return rows

    def find_frames(self, limit=50, **filters):
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        rows = self._rows(**filters)
        return dict(
            status="ok" if rows else "no_frames_in_range",
            collection_id=self.collection_id,
            matched_frame_count=len(rows),
            returned=min(limit, len(rows)),
            truncated=len(rows) > limit,
            frames=rows[:limit],
            caveat="YOLO labels are evidence; missing labels are not proof of absence.",
        )

    def search(
        self,
        query,
        target_label=None,
        top_k=5,
        min_similarity=None,
        group_results=True,
        group_window_s=2.0,
        **filters,
    ):
        if not str(query).strip() or not 1 <= top_k <= 50:
            raise ValueError("query is required; top_k must be 1..50")
        if not math.isfinite(group_window_s) or group_window_s < 0:
            raise ValueError("invalid group window")
        if min_similarity is not None and (
            not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1
        ):
            raise ValueError("invalid similarity floor")
        all_rows = self._rows(**filters)
        rows = [r for r in all_rows if r["radio_status"] == "ready"]
        target = verification_target(query, target_label)
        result = dict(
            status="no_frames_in_range",
            query=query,
            matches=[],
            threshold_calibrated=False,
            indexed_frame_count_in_range=len(rows),
            pending_frame_count_in_range=sum(
                r["radio_status"] == "pending" for r in all_rows
            ),
            retrieval_strategy="radio_then_yolo"
            if target in COCO_CLASSES
            else "radio_only",
            verification_target=target,
            calibration_note="Cosine similarity is not a probability. Return ranked candidates, not confirmed presence.",
        )
        if not rows:
            return result
        store = RecordStore.open_readonly(self.root, self.collection_id)
        try:
            if store.model_fingerprint != self.runtime.fingerprint:
                raise RuntimeError(
                    "query encoder fingerprint differs from indexed collection"
                )
            frames = [store.get_frame(r["frame_id"]) for r in rows]
            matrix = store.load_vectors(frames)
        finally:
            store.close()
        scores = matrix @ self.runtime.encode_text(query)
        selected = []
        for i in np.argsort(-scores):
            row = rows[int(i)].copy()
            score = float(scores[i])
            if min_similarity is not None and score < min_similarity:
                continue
            if group_results and any(
                r["camera_id"] == row["camera_id"]
                and r.get("coordinate_id") == row.get("coordinate_id")
                and r.get("session_id") == row.get("session_id")
                and abs(
                    (
                        parse_time(r["captured_at"]) - parse_time(row["captured_at"])
                    ).total_seconds()
                )
                <= group_window_s
                for r in selected
            ):
                continue
            row["similarity"] = round(score, 4)
            ds = [d for d in row["detections"] if d["label"] == target]
            state = (
                "not_supported"
                if target not in COCO_CLASSES
                else "unavailable"
                if row.get("detection_status") != "available"
                else "supported_by_yolo"
                if ds
                else "not_detected"
            )
            row["yolo_verification"] = dict(
                status=state,
                target_label=target,
                detections=ds,
                note="Class evidence only; attributes and relationships are not verified. Non-detection is not proof of absence.",
            )
            selected.append(row)
            if len(selected) >= top_k:
                break
        result.update(
            status="ok" if selected else "no_match_above_threshold", matches=selected
        )
        return result

    def get_frame(self, frame_id):
        with connect(self.root) as db:
            row = db.execute(
                "SELECT * FROM observations WHERE frame_id=? AND collection_id=?",
                (frame_id, self.collection_id),
            ).fetchone()
            if not row:
                return dict(status="not_found", frame_id=frame_id)
            result = dict(status="ok", **payload(row))
            vector = db.execute(
                "SELECT f.camera_id AS stream_id,f.vector_partition AS partition,f.vector_row AS row,c.embedding_dim AS dim "
                "FROM frames f JOIN collections c ON c.collection_id=f.collection_id "
                "WHERE f.frame_id=? AND f.collection_id=?",
                (frame_id, self.collection_id),
            ).fetchone()
            result["vector"] = (
                dict(**dict(vector), dtype_on_disk="float16") if vector else None
            )
            return result

    def search_evidence(self, *, include_images=True, max_edge=1280, **arguments):
        result = self.search(**arguments)
        result["candidate_only"] = True
        return attach_visual_evidence(
            result,
            result["matches"],
            lambda row, edge: self.image(row["frame_id"], max_edge=edge),
            include_images=include_images,
            max_edge=max_edge,
        )

    def image(self, frame_id, overlay=False, max_edge=1280):
        if not 64 <= max_edge <= 4096:
            raise ValueError("max_edge must be 64..4096")
        m = self.get_frame(frame_id)
        if not m.get("image_path"):
            raise FileNotFoundError("frame has no stored image")
        path = (self.root / m["image_path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("image path escapes record root")
        return render_jpeg(path, max_edge, m["detections"] if overlay else None)

    def compare_frames(self, frame_ids):
        if not 2 <= len(frame_ids) <= 50:
            raise ValueError("compare 2..50 frames")
        rows = [self.get_frame(f) for f in frame_ids]
        if any(r["status"] != "ok" for r in rows):
            return dict(status="not_found", frames=rows)
        first = rows[0]
        for r in rows:
            a, b = first.get("camera_pose"), r.get("camera_pose")
            valid = (
                a is not None
                and b is not None
                and first.get("coordinate_id") is not None
                and first.get("coordinate_id") == r.get("coordinate_id")
                and first.get("session_id") == r.get("session_id")
                and a["frame"] == b["frame"]
            )
            r["distance_from_first_m"] = (
                math.dist(
                    [a[k] for k in ("x", "y", "z")], [b[k] for k in ("x", "y", "z")]
                )
                if valid
                else None
            )
            r["rotation_from_first_deg"] = (
                rotation_deg(
                    [a[k] for k in ("qx", "qy", "qz", "qw")],
                    [b[k] for k in ("qx", "qy", "qz", "qw")],
                )
                if valid
                else None
            )
        return dict(
            status="ok",
            frames=rows,
            note="Positions locate the observing camera, not the object. Unrelated coordinate sessions are not comparable.",
        )

    def describe(
        self,
        mode="current",
        location_label=None,
        camera_id=None,
        device_id=None,
        max_age_s=5,
        place_id=None,
        _snapshots=None,
    ):
        freshness(dt.datetime.now(dt.timezone.utc).isoformat(), max_age_s)
        if device_id is not None:
            raise ValueError("backend owns no camera; filter by logical camera_id")
        if mode == "current":
            if location_label or place_id:
                raise ValueError("use location mode for a place")
            with self.runtime.lock:
                if _snapshots is not None:
                    _snapshots.update(
                        {
                            m["frame_id"]: jpeg
                            for cid, (m, jpeg) in self.runtime.live.items()
                            if camera_id is None or cid == camera_id
                        }
                    )
                rows = [
                    dict(m)
                    for cid, (m, _) in self.runtime.live.items()
                    if camera_id is None or cid == camera_id
                ]
        elif mode == "location":
            if place_id is None:
                places = [
                    p
                    for p in self.list_places()["places"]
                    if p["name"] == location_label
                ]
                if len(places) != 1:
                    return dict(
                        status="location_not_found"
                        if not places
                        else "ambiguous_location",
                        places=places,
                        views=[],
                        radio_used=False,
                    )
                place_id = places[0]["place_id"]
            rows = self._rows(
                place_id=place_id,
                camera_ids=[camera_id] if camera_id else None,
                start_time="1970-01-01T00:00:00+00:00",
            )
            latest = {}
            for r in rows:
                latest[r["camera_id"]] = r
            rows = list(latest.values())
        else:
            raise ValueError("mode must be current or location")
        for r in rows:
            r.update(freshness(r["captured_at"], max_age_s))
            r["pose"] = r.get("camera_pose")
            counts = {}
            for d in r.get("detections", []):
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            r["label_counts"] = (
                counts if r.get("detection_status") == "available" else None
            )
        status = (
            "no_observations"
            if not rows
            else "stale"
            if any(r["stale"] for r in rows)
            else "detection_unavailable"
            if any(r.get("detection_status") != "available" for r in rows)
            else "ok"
        )
        return dict(
            status=status,
            mode=mode,
            place_id=place_id,
            location_label=location_label,
            views=rows,
            radio_used=False,
            source="live_cache" if mode == "current" else "latest_record",
            caveat="YOLO class inventory within its ROI. Counts are per view, not unique objects. Historical place observations do not describe the present.",
        )

    def describe_evidence(self, *, include_images=True, max_edge=1280, **arguments):
        # Snapshot JPEG and metadata in the same lock. A current observation may
        # have been skipped by keyframe sampling and have no historical file.
        snapshots = {}
        result = self.describe(
            _snapshots=snapshots if include_images else None, **arguments
        )

        def load(row, edge):
            if result["mode"] == "current":
                return render_jpeg(io.BytesIO(snapshots[row["frame_id"]]), edge)
            return self.image(row["frame_id"], max_edge=edge)

        return attach_visual_evidence(
            result,
            result["views"],
            load,
            include_images=include_images,
            max_edge=max_edge,
        )

    def status(self):
        return self.runtime.health()
