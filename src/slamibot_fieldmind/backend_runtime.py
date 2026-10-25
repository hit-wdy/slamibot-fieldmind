"""Persistent runtime. One database writer and one shared inference scheduler."""

from __future__ import annotations

import concurrent.futures
import math
import queue
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from .keyframes import KeyframePolicy
from .protocol import identifier, timestamp, utcnow, validate_observation
from .semantic_store import ObservationStore
from .backend_lock import BackendLock


@dataclass
class BackendConfig:
    root: Path = Path("/mnt/usb/slamibot_record_data/semantic_index")
    collection_id: str = "radio_clip_v1"
    min_free_bytes: int = 2 * 1024**3
    pending_limit: int = 32
    inference_queue_size: int = 4
    require_mount: bool = True
    policy: KeyframePolicy = field(default_factory=KeyframePolicy)

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        identifier(self.collection_id, "collection_id")
        if (
            self.min_free_bytes < 0
            or self.pending_limit < 1
            or not 1 <= self.inference_queue_size <= 4
        ):
            raise ValueError("invalid backend capacity")


class Writer:
    def __init__(self, root, collection):
        self.commands = queue.Queue(maxsize=64)
        self.ready = concurrent.futures.Future()
        self.thread = threading.Thread(
            target=self._run,
            args=(root, collection),
            name="semantic-writer",
            daemon=True,
        )
        self.thread.start()
        self.ready.result(timeout=30)

    def _run(self, root, collection):
        try:
            store = ObservationStore(root, collection)
        except Exception as e:
            self.ready.set_exception(e)
            return
        self.ready.set_result(True)
        try:
            while True:
                item = self.commands.get()
                if item is None:
                    break
                fn, future = item
                try:
                    future.set_result(fn(store))
                except Exception as e:
                    future.set_exception(e)
        finally:
            store.close()

    def call(self, fn):
        future = concurrent.futures.Future()
        self.commands.put((fn, future), timeout=10)
        return future.result(timeout=120)

    def close(self):
        self.commands.put(None, timeout=10)
        self.thread.join(timeout=30)


class BackendRuntime:
    def __init__(self, config=None, encoder_factory=None):
        self.config = config or BackendConfig()
        if self.config.require_mount and not Path("/mnt/usb").is_mount():
            raise RuntimeError(
                "/mnt/usb must be mounted; refusing to write to the root filesystem"
            )
        self.config.root.mkdir(parents=True, exist_ok=True)
        self.owner_lock = BackendLock(self.config.root)
        try:
            self.writer = Writer(self.config.root, self.config.collection_id)
        except Exception:
            self.owner_lock.close()
            raise
        self.encoder_factory = encoder_factory
        self.encoder = None
        self.model_state = "loading"
        self.model_error = None
        self.lock = threading.RLock()
        self.live = {}
        self.closed = threading.Event()
        self.wake = threading.Event()
        self.text_jobs = queue.Queue(maxsize=8)
        self.metrics = dict(
            received=0,
            accepted=0,
            duplicates=0,
            skipped=0,
            overloaded=0,
            low_disk=0,
            failed=0,
            overdue_gaps=0,
            stationary_skipped=0,
            time_only_accepted=0,
        )
        self.latest_selection = None
        self.sampling_blocked = {}
        self.fingerprint = None
        self.latencies = []
        self.thread = threading.Thread(
            target=self._supervise_inference, name="radio-inference", daemon=True
        )
        self.thread.start()

    def _supervise_inference(self):
        try:
            self._infer()
        except Exception as error:
            # A storage/worker failure must not leave a dead thread marked ready.
            self.model_state = "failed"
            self.model_error = str(error)
            while not self.closed.is_set():
                try:
                    _, future = self.text_jobs.get(timeout=0.1)
                    future.set_exception(RuntimeError(self.model_error))
                except queue.Empty:
                    pass

    def _infer(self):
        try:
            if self.encoder_factory is None:
                from .encoder import AlignedEncoder

                self.encoder_factory = AlignedEncoder
            self.encoder = self.encoder_factory()
            self.fingerprint = self.encoder.fingerprint()
            self.writer.call(lambda s: s.bind_encoder(self.encoder))
            self.model_state = "ready"
        except Exception as error:
            self.model_error = str(error)
            self.model_state = "failed"
        while not self.closed.is_set():
            # At most one text request between image jobs: neither stream starves.
            try:
                text, future = self.text_jobs.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    if self.model_state != "ready":
                        raise RuntimeError(self.model_error or self.model_state)
                    future.set_result(self.encoder.encode_texts([text])[0])
                except Exception as error:
                    future.set_exception(error)
            jobs = (
                self.writer.call(lambda s: s.pending(self.config.inference_queue_size))
                if self.model_state == "ready"
                else []
            )
            if jobs:
                # Only one job is decoded at a time; pending JPEGs stay on disk.
                m = jobs[0]
                try:
                    started = time.monotonic()
                    with Image.open(self.config.root / m["image_path"]) as image:
                        vector = self.encoder.encode_images([image.convert("RGB")])[0]
                    self.writer.call(lambda s: s.complete(m, vector, self.fingerprint))
                    from .query import parse_time

                    with self.lock:
                        self.latencies.append(
                            dict(
                                inference_ms=(time.monotonic() - started) * 1000,
                                ready_after_accept_s=max(
                                    0,
                                    (
                                        parse_time(utcnow())
                                        - parse_time(
                                            m.get("accepted_at", m["received_at"])
                                        )
                                    ).total_seconds(),
                                ),
                            )
                        )
                        self.latencies = self.latencies[-3000:]
                except Exception as error:
                    error_message = str(error)
                    self.writer.call(lambda s: s.failed(m["frame_id"], error_message))
                    with self.lock:
                        self.metrics["failed"] += 1
            else:
                self.wake.wait(0.1)
                self.wake.clear()
        while True:
            try:
                _, future = self.text_jobs.get_nowait()
                future.set_exception(RuntimeError("backend stopped"))
            except queue.Empty:
                break

    def encode_text(self, text):
        if self.closed.is_set():
            raise RuntimeError("backend stopped")
        future = concurrent.futures.Future()
        self.text_jobs.put((text, future), timeout=5)
        self.wake.set()
        return future.result(timeout=120)

    def session(self, raw):
        m = dict(raw)
        for k in ("session_id", "coordinate_id", "task_id"):
            identifier(m.get(k), k)
        if m.get("state") not in {
            "starting",
            "running",
            "stopping",
            "stopped",
            "fault",
        }:
            raise ValueError("invalid session state")
        if m.get("started_at"):
            m["started_at"] = timestamp(m["started_at"])

        def write(s):
            with s.db:
                if m["state"] in {"starting", "running"}:
                    s.db.execute(
                        "UPDATE sessions SET state='stopped',updated_at=? WHERE session_id!=? AND state IN ('starting','running')",
                        (utcnow(), m["session_id"]),
                    )
            s.session(m)
            return dict(status="ok", **m)

        return self.writer.call(write)

    def ingest(self, raw, jpeg):
        m = validate_observation(raw, jpeg)
        with self.lock:
            self.metrics["received"] += 1
            old = self.live.get(m["camera_id"])
            newer_sequence = (
                old is None
                or m["source_id"] != old[0]["source_id"]
                or m["source_sequence"] > old[0]["source_sequence"]
            )
            if old is None or (
                newer_sequence and m["received_at"] >= old[0]["received_at"]
            ):
                self.live[m["camera_id"]] = (dict(m), jpeg)
                if len(self.live) > 8:
                    del self.live[next(iter(self.live))]

        def write(s):
            existing = s.get_observation(m["observation_key"])
            if existing:
                return "duplicates", existing
            session = s.db.execute(
                "SELECT * FROM sessions WHERE session_id=?", (m.get("session_id"),)
            ).fetchone()
            if (
                not session
                or session["state"] != "running"
                or session["coordinate_id"] != m.get("coordinate_id")
            ):
                return "skipped", dict(reason="no_active_coordinate_session")
            m["task_id"] = session["task_id"]
            last = s.db.execute(
                "SELECT metadata_json FROM observations WHERE collection_id=? AND session_id=? AND coordinate_id=? AND camera_id=? ORDER BY source_stamp_ns DESC LIMIT 1",
                (s.collection_id, m["session_id"], m["coordinate_id"], m["camera_id"]),
            ).fetchone()
            import json

            decision = self.config.policy.decide(
                m["source_stamp_ns"],
                m.get("camera_pose"),
                json.loads(last[0]) if last else None,
                body_pose=m.get("body_pose"),
            )
            m["selection"] = decision
            if not decision["reasons"]:
                return "skipped", dict(reason="keyframe_policy", selection=decision)
            count = s.db.execute(
                "SELECT COUNT(*) FROM observations WHERE collection_id=? AND radio_status='pending'",
                (s.collection_id,),
            ).fetchone()[0]
            if count >= self.config.pending_limit:
                return "overloaded", dict(
                    reason="pending_limit", pending=count, selection=decision
                )
            if shutil.disk_usage(self.config.root).free < self.config.min_free_bytes:
                return "low_disk", dict(
                    reason="disk_free_below_limit", selection=decision
                )
            m["accepted_at"] = utcnow()
            return "accepted", s.accept(m, jpeg)

        kind, result = self.writer.call(write)
        with self.lock:
            self.metrics[kind] += 1
            selection_status = result.get("selection", {}).get("status")
            if kind == "skipped" and selection_status == "stationary":
                self.metrics["stationary_skipped"] += 1
            if kind == "accepted" and selection_status == "pose_unavailable":
                self.metrics["time_only_accepted"] += 1
            if kind in {"overloaded", "low_disk"}:
                self.sampling_blocked[m["camera_id"]] = True
                if len(self.sampling_blocked) > 8:
                    del self.sampling_blocked[next(iter(self.sampling_blocked))]
            if kind == "accepted":
                self.latest_selection = utcnow()
                if (
                    self.sampling_blocked.pop(m["camera_id"], False)
                    and (result.get("selection", {}).get("interval_s") or 0)
                    > self.config.policy.max_interval_s
                    + self.config.policy.min_interval_s
                ):
                    self.metrics["overdue_gaps"] += 1
        self.wake.set()
        return {"status": kind, **result}

    def put_place(self, raw):
        p = dict(raw)
        for k in ("place_id", "coordinate_id"):
            identifier(p.get(k), k)
        if not str(p.get("name", "")).strip() or not str(p.get("frame", "")).strip():
            raise ValueError("place requires name and coordinate frame")
        for k in ("x", "y", "z", "radius_m"):
            p[k] = float(p.get(k, 3 if k == "radius_m" else 0))
            if not math.isfinite(p[k]):
                raise ValueError("place must be finite")
        if p["radius_m"] <= 0:
            raise ValueError("radius must be positive")

        def write(s):
            with s.db:
                s.db.execute(
                    "INSERT INTO places VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(place_id) DO UPDATE SET name=excluded.name,coordinate_id=excluded.coordinate_id,frame=excluded.frame,x=excluded.x,y=excluded.y,z=excluded.z,radius_m=excluded.radius_m",
                    tuple(
                        p[k]
                        for k in (
                            "place_id",
                            "name",
                            "coordinate_id",
                            "frame",
                            "x",
                            "y",
                            "z",
                            "radius_m",
                        )
                    ),
                )
            return p

        return self.writer.call(write)

    def health(self):
        def read(s):
            states = {
                r[0]: r[1]
                for r in s.db.execute(
                    "SELECT radio_status,COUNT(*) FROM observations WHERE collection_id=? GROUP BY radio_status",
                    (s.collection_id,),
                )
            }
            coverage = [
                dict(r)
                for r in s.db.execute(
                    "SELECT camera_id,COUNT(*) AS frames,MIN(captured_at) AS first,MAX(captured_at) AS last FROM observations WHERE collection_id=? GROUP BY camera_id",
                    (s.collection_id,),
                )
            ]
            latest = s.db.execute(
                "SELECT MAX(captured_at) FROM observations WHERE collection_id=?",
                (s.collection_id,),
            ).fetchone()[0]
            integrity = (
                s.vectors.verify()
                if s.vectors
                else {"ok": None, "orphan_or_missing": []}
            )
            return states, coverage, latest, integrity

        states, coverage, latest, integrity = self.writer.call(read)
        from .backend_query import freshness

        with self.lock:
            live = {
                k: {
                    **freshness(m["captured_at"], 5),
                    "detection_status": m["detection_status"],
                    "sync_quality": m["sync_quality"],
                    "pose_available": m.get("camera_pose") is not None,
                }
                for k, (m, _) in self.live.items()
            }
            metrics = dict(self.metrics)
            timings = {
                key: sorted(row[key] for row in self.latencies)
                for key in ("inference_ms", "ready_after_accept_s")
            }
            percentiles = {
                key: {
                    "samples": len(values),
                    "p50": values[int((len(values) - 1) * 0.5)] if values else None,
                    "p95": values[int((len(values) - 1) * 0.95)] if values else None,
                }
                for key, values in timings.items()
            }
        healthy = (
            self.model_state == "ready"
            and bool(live)
            and not any(v["stale"] for v in live.values())
        )
        return dict(
            status="ok" if healthy else "degraded",
            model_state=self.model_state,
            model_error=self.model_error,
            model_fingerprint=self.fingerprint,
            collection_id=self.config.collection_id,
            states=states,
            indexed_frames=states.get("ready", 0),
            cameras=coverage,
            latest_record_at=latest,
            live=live,
            counters=metrics,
            latest_keyframe_accepted_at=self.latest_selection,
            disk_free_bytes=shutil.disk_usage(self.config.root).free,
            pending_limit=self.config.pending_limit,
            keyframe_policy=asdict(self.config.policy),
            latency=percentiles,
            vector_partitions_ok=integrity["ok"],
            integrity_problems=integrity["orphan_or_missing"],
            stores_images=True,
            threshold_calibrated=False,
        )

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.wake.set()
        self.thread.join(timeout=180)
        if self.thread.is_alive():
            raise RuntimeError(
                "inference did not stop; writer remains open to protect pending work"
            )
        self.writer.close()
        self.owner_lock.close()
