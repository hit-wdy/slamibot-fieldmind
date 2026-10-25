"""Additive, versioned observation journal beside the legacy vector index.

Only the backend writer thread calls mutations. Pending observations live outside
the legacy frames table, so older readers only ever see complete vector records.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .detection_types import Detection
from .pose import Pose
from .protocol import observation_key, utcnow
from .store import RecordStore, SCHEMA, SCHEMA_VERSION

SEMANTIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_schema(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
 session_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, state TEXT NOT NULL,
 coordinate_id TEXT NOT NULL, started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observations(
 frame_id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, session_id TEXT,
 coordinate_id TEXT, task_id TEXT, camera_id TEXT NOT NULL, captured_at TEXT NOT NULL,
 source_stamp_ns INTEGER NOT NULL, image_path TEXT, radio_status TEXT NOT NULL,
 radio_error TEXT, model_fingerprint TEXT, metadata_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS obs_time ON observations(collection_id,captured_at);
CREATE INDEX IF NOT EXISTS obs_session ON observations(session_id,coordinate_id,camera_id,source_stamp_ns);
CREATE INDEX IF NOT EXISTS obs_status ON observations(collection_id,radio_status);
CREATE TABLE IF NOT EXISTS places(
 place_id TEXT PRIMARY KEY, name TEXT NOT NULL, coordinate_id TEXT NOT NULL,
 frame TEXT NOT NULL, x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,
 radius_m REAL NOT NULL);
"""


@contextmanager
def connect(root):
    connection = sqlite3.connect(
        f"{(Path(root) / 'records.sqlite').as_uri()}?mode=ro", uri=True, timeout=10
    )
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def payload(row):
    result = json.loads(row["metadata_json"])
    for key in (
        "frame_id",
        "session_id",
        "coordinate_id",
        "task_id",
        "image_path",
        "radio_status",
        "radio_error",
        "model_fingerprint",
    ):
        result[key] = row[key]
    result["pose"] = result.get("camera_pose")
    if result["pose"] and result["pose"].get("source") != "slam":
        result["pose_is_a_stand_in"] = (
            "This is an offline/simulated pose, not a measured camera location."
        )
    result["image_url"] = (
        f"/v1/frames/{row['frame_id']}/image" if row["image_path"] else None
    )
    counts = {}
    for d in result.get("detections", []):
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    result["label_counts"] = (
        counts if result.get("detection_status") == "available" else None
    )
    return result


class ObservationStore:
    def __init__(self, root, collection_id):
        self.root = Path(root).resolve()
        self.collection_id = collection_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "records.sqlite", timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        has_frames = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='frames'"
        ).fetchone()
        if has_frames and version != SCHEMA_VERSION:
            self.db.close()
            raise RuntimeError(
                f"unsupported legacy schema version {version}; no data was migrated"
            )
        self.db.executescript(SCHEMA + SEMANTIC_SCHEMA)
        row = self.db.execute("SELECT version FROM semantic_schema").fetchone()
        if row and row[0] not in (1, 2):
            self.db.close()
            raise RuntimeError("unsupported semantic schema version")
        # Version 2 separates retry identity from human-readable frame IDs.
        # Backfill incrementally; never rename historical images or vectors.
        if not row or row[0] == 1:
            self.db.execute("BEGIN IMMEDIATE")
            with self.db:
                self.db.execute(
                    "ALTER TABLE observations ADD COLUMN observation_key TEXT"
                )
                for old in self.db.execute(
                    "SELECT frame_id,metadata_json FROM observations"
                ):
                    m = json.loads(old["metadata_json"])
                    if all(
                        k in m for k in ("device_id", "source_id", "source_sequence")
                    ):
                        self.db.execute(
                            "UPDATE observations SET observation_key=? WHERE frame_id=?",
                            (observation_key(m), old["frame_id"]),
                        )
                self.db.execute(
                    "CREATE UNIQUE INDEX obs_identity ON observations(collection_id,observation_key)"
                )
                if row:
                    self.db.execute("UPDATE semantic_schema SET version=2")
                else:
                    self.db.execute("INSERT INTO semantic_schema VALUES (2)")
        self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.db.commit()
        self.vectors = None
        # Add journal entries for older complete records without touching their vectors.
        for row in self.db.execute(
            "SELECT * FROM frames WHERE collection_id=? AND frame_id NOT IN (SELECT frame_id FROM observations)",
            (collection_id,),
        ).fetchall():
            ds = [
                dict(
                    label=d["label"],
                    label_id=d["label_id"],
                    confidence=d["confidence"],
                    bbox_norm=[d["x1"], d["y1"], d["x2"], d["y2"]],
                )
                for d in self.db.execute(
                    "SELECT * FROM detections WHERE frame_id=?", (row["frame_id"],)
                )
            ]
            p = {
                k: row["pose_" + k]
                for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "frame", "source")
            }
            legacy = dict(
                frame_id=row["frame_id"],
                camera_id=row["camera_id"],
                captured_at=row["captured_at"],
                received_at=row["captured_at"],
                source_stamp_ns=0,
                camera_pose=p,
                body_pose=None,
                detections=ds,
                detection_status="unknown",
                timestamp_source=row["timestamp_source"],
                sync_quality="unknown",
                legacy=True,
                session_id=None,
                coordinate_id=None,
                task_id=None,
            )
            self._insert(legacy, None, "ready")
        self.db.commit()

    def _insert(self, m, image_path, status):
        self.db.execute(
            "INSERT INTO observations(frame_id,collection_id,session_id,coordinate_id,task_id,camera_id,captured_at,source_stamp_ns,image_path,radio_status,metadata_json,observation_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                m["frame_id"],
                self.collection_id,
                m.get("session_id"),
                m.get("coordinate_id"),
                m.get("task_id"),
                m["camera_id"],
                m["captured_at"],
                m["source_stamp_ns"],
                image_path,
                status,
                json.dumps(m, ensure_ascii=False, allow_nan=False),
                m.get("observation_key"),
            ),
        )

    def get(self, frame_id):
        row = self.db.execute(
            "SELECT * FROM observations WHERE frame_id=? AND collection_id=?",
            (frame_id, self.collection_id),
        ).fetchone()
        return payload(row) if row else None

    def accept(self, m, jpeg):
        existing = self.get_observation(m["observation_key"])
        if existing:
            return existing
        path = self.root / "images" / m["frame_id"][:5] / (m["frame_id"] + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as f:
            f.write(jpeg)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
        with self.db:
            self._insert(m, path.relative_to(self.root).as_posix(), "pending")
        return self.get(m["frame_id"])

    def get_observation(self, key):
        row = self.db.execute(
            "SELECT * FROM observations WHERE observation_key=? AND collection_id=?",
            (key, self.collection_id),
        ).fetchone()
        return payload(row) if row else None

    def bind_encoder(self, encoder):
        if self.vectors is None:
            self.vectors = RecordStore(
                self.root,
                self.collection_id,
                encoder.fingerprint(),
                encoder.embedding_dim,
            ).open()

    def pending(self, limit=4):
        return [
            payload(r)
            for r in self.db.execute(
                "SELECT * FROM observations WHERE collection_id=? AND radio_status='pending' ORDER BY captured_at LIMIT ?",
                (self.collection_id, limit),
            )
        ]

    def complete(self, m, vector, fingerprint):
        if not self.vectors:
            raise RuntimeError("encoder is not bound")
        # A crash after legacy commit but before journal commit is recoverable.
        if self.vectors.get_frame(m["frame_id"]) is None:
            p = dict(m["camera_pose"]) if m.get("camera_pose") else None
            if p:
                # Legacy readers do not know coordinate_id; scope their frame too.
                p["frame"] = p["frame"] + "@" + m["coordinate_id"]
            pose = (
                Pose(**p)
                if p
                else Pose(
                    0, 0, 0, 0, 0, 0, 1, frame="unavailable", source="unavailable"
                )
            )
            # The legacy UNIQUE(camera,time,sequence) predates SLAM sessions.
            # Give each producer/session a storage stream; retain the physical
            # camera_id in observation metadata and all public backend results.
            stream_key = [
                m.get(k)
                for k in (
                    "device_id",
                    "camera_id",
                    "session_id",
                    "coordinate_id",
                    "source_id",
                )
            ]
            stream_id = (
                "ros-" + hashlib.sha256(repr(stream_key).encode()).hexdigest()[:24]
            )
            self.vectors.register_camera(
                stream_id,
                source_type="ros",
                pose_frame=pose.frame,
                pose_source=pose.source,
            )
            self.vectors.add(
                stream_id,
                dt.datetime.fromisoformat(m["captured_at"]),
                m["source_sequence"],
                vector,
                pose,
                [Detection(**d) for d in m["detections"]],
                m["timestamp_source"],
                m.get("device_timestamp_ns"),
                frame_id=m["frame_id"],
            )
        with self.db:
            self.db.execute(
                "UPDATE observations SET radio_status='ready',radio_error=NULL,model_fingerprint=? WHERE frame_id=?",
                (fingerprint, m["frame_id"]),
            )

    def failed(self, frame_id, error):
        with self.db:
            self.db.execute(
                "UPDATE observations SET radio_status='failed',radio_error=? WHERE frame_id=?",
                (str(error), frame_id),
            )

    def session(self, m):
        with self.db:
            self.db.execute(
                "INSERT INTO sessions VALUES(?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET state=excluded.state,coordinate_id=excluded.coordinate_id,updated_at=excluded.updated_at,metadata_json=excluded.metadata_json",
                (
                    m["session_id"],
                    m["task_id"],
                    m["state"],
                    m["coordinate_id"],
                    m.get("started_at", utcnow()),
                    utcnow(),
                    json.dumps(m),
                ),
            )

    def close(self):
        if self.vectors:
            self.vectors.close()
        self.db.close()
