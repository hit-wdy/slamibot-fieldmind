"""Persistent records: embedding + detections + pose, with no images.

Per the requirement, frames are not stored. A record holds the frame identity,
the RADIO embedding, the YOLO detections and the camera pose for one sampled
instant.

Layout, following spec section 6:

    <root>/records.sqlite                                     metadata
    <root>/vectors/<collection>/<camera>/<YYYYMMDDTHH>.f16     raw float16 rows
    <root>/vectors/<collection>/<camera>/<YYYYMMDDTHH>.json    partition sidecar

Vectors are appended to hour-sized partitions rather than one file per frame, as
the spec requires, and addressed by `(partition, row)` from the database. The
database is the source of truth: a row in `frames` exists only after its vector
bytes are on disk, so a present row means a complete record. That is why there is
no `status` column yet -- the asynchronous queue that would need one is section 5
and is not implemented. `verify()` re-checks the two stores against each other.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .detection_types import Detection
from .pose import Pose

DTYPE = np.float16
DTYPE_SIZE = np.dtype(DTYPE).itemsize

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    collection_id     TEXT PRIMARY KEY,
    model_fingerprint TEXT NOT NULL,
    embedding_dim     INTEGER NOT NULL,
    configuration_json TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id      TEXT PRIMARY KEY,
    location_label TEXT,
    source_type    TEXT NOT NULL,
    pose_frame     TEXT,
    pose_source    TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    frame_id         TEXT PRIMARY KEY,
    camera_id        TEXT NOT NULL,
    captured_at      TEXT NOT NULL,
    timestamp_source TEXT NOT NULL,
    source_sequence  INTEGER NOT NULL,
    -- Monotonic device clock, nanoseconds. This is the key a SLAM front end can
    -- correlate a pose against; wall-clock `captured_at` cannot be aligned to a
    -- SLAM timebase, and without this the pose could only be guessed at.
    device_timestamp_ns INTEGER,
    ingested_at      TEXT NOT NULL,
    collection_id    TEXT NOT NULL REFERENCES collections(collection_id),
    vector_partition TEXT NOT NULL,
    vector_row       INTEGER NOT NULL,
    -- A record always carries a pose; `pose_source` says whether it was
    -- measured or simulated. NULL here would turn into a None that only blows
    -- up later, inside arithmetic.
    pose_x REAL NOT NULL, pose_y REAL NOT NULL, pose_z REAL NOT NULL,
    pose_qx REAL NOT NULL, pose_qy REAL NOT NULL, pose_qz REAL NOT NULL, pose_qw REAL NOT NULL,
    pose_frame TEXT NOT NULL,
    pose_source TEXT NOT NULL,
    detection_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (camera_id, captured_at, source_sequence)
);

CREATE TABLE IF NOT EXISTS detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id   TEXT NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    label_id   INTEGER NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_camera_time  ON frames(camera_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_frames_collection   ON frames(collection_id);
CREATE INDEX IF NOT EXISTS idx_frames_pose_xyz     ON frames(pose_x, pose_y, pose_z);
CREATE INDEX IF NOT EXISTS idx_detections_frame    ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_label    ON detections(label);
"""


@dataclass(frozen=True)
class StoredFrame:
    frame_id: str
    camera_id: str
    captured_at: dt.datetime
    source_sequence: int
    pose: Pose
    detections: tuple[Detection, ...]
    vector_partition: str
    vector_row: int
    device_timestamp_ns: int | None = None

    @property
    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
        return counts


def make_frame_id(camera_id: str, captured_at: dt.datetime, source_sequence: int) -> str:
    """A stable, human-readable id. Deliberately not an array index."""
    stamp = captured_at.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3]
    return f"{camera_id}-{stamp}-{source_sequence:06d}"


def partition_for(captured_at: dt.datetime) -> str:
    return captured_at.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H")


class RecordStore:
    """SQLite metadata plus append-only float16 vector partitions."""

    def __init__(
        self,
        root: Path | str,
        collection_id: str,
        model_fingerprint: str,
        embedding_dim: int,
        configuration: dict | None = None,
    ) -> None:
        self.root = Path(root)
        self.collection_id = collection_id
        self.model_fingerprint = model_fingerprint
        self.embedding_dim = int(embedding_dim)
        self.configuration = configuration or {}
        self.db_path = self.root / "records.sqlite"
        self.vectors_root = self.root / "vectors" / collection_id
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def open_existing(cls, root: Path | str, collection_id: str) -> "RecordStore":
        """Open a collection, taking the fingerprint and dimension from the database.

        A reader has no reason to know them in advance, and demanding them would
        mean every read path had to load the model first.
        """
        root = Path(root)
        db_path = root / "records.sqlite"
        if not db_path.is_file():
            raise FileNotFoundError(
                f"{db_path} does not exist; record something first with scripts/record_live.py"
            )
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT model_fingerprint, embedding_dim FROM collections WHERE collection_id = ?",
                (collection_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise KeyError(f"collection {collection_id!r} is not in {db_path}")
        return cls(root, collection_id, row[0], int(row[1])).open()

    # -- lifecycle --------------------------------------------------------------

    @classmethod
    def open_readonly(cls, root: Path | str, collection_id: str) -> "RecordStore":
        """Reader for the persistent backend; never executes schema/write SQL."""
        root = Path(root).resolve()
        connection = sqlite3.connect((root / "records.sqlite").as_uri() + "?mode=ro", uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT model_fingerprint,embedding_dim FROM collections WHERE collection_id=?", (collection_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"collection {collection_id!r} has no bound encoder yet")
            result = cls(root, collection_id, row[0], int(row[1]))
            result._connection = connection
            return result
        except Exception:
            connection.close()
            raise

    def open(self) -> "RecordStore":
        if self._connection is not None:
            return self
        self.root.mkdir(parents=True, exist_ok=True)
        self.vectors_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        # `user_version` alone cannot tell a fresh file from a pre-versioning one:
        # both report 0. The presence of the table is what distinguishes them.
        has_tables = (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='frames'"
            ).fetchone()[0]
            > 0
        )
        existing_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if has_tables and existing_version != SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.db_path} holds a schema version {existing_version} collection, "
                f"this build writes version {SCHEMA_VERSION}. There is no migration yet: "
                "point --out at a new directory, or rebuild the collection."
            )
        connection.executescript(SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.execute(
            "INSERT OR IGNORE INTO collections "
            "(collection_id, model_fingerprint, embedding_dim, configuration_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                self.collection_id,
                self.model_fingerprint,
                self.embedding_dim,
                json.dumps(self.configuration, sort_keys=True),
                dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        existing = connection.execute(
            "SELECT model_fingerprint, embedding_dim FROM collections WHERE collection_id = ?",
            (self.collection_id,),
        ).fetchone()
        if existing["model_fingerprint"] != self.model_fingerprint:
            raise RuntimeError(
                f"collection {self.collection_id!r} already holds vectors from a different space.\n"
                f"  stored   {existing['model_fingerprint']}\n"
                f"  current  {self.model_fingerprint}\n"
                "Use a new collection, or re-encode: vectors from different fingerprints are not comparable."
            )
        if int(existing["embedding_dim"]) != self.embedding_dim:
            raise RuntimeError(
                f"collection {self.collection_id!r} has dim {existing['embedding_dim']}, "
                f"this encoder produces {self.embedding_dim}"
            )
        connection.commit()
        self._connection = connection
        return self

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "RecordStore":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("RecordStore is not open")
        return self._connection

    # -- write ------------------------------------------------------------------

    def register_camera(
        self,
        camera_id: str,
        source_type: str = "oak_camera",
        location_label: str | None = None,
        pose_frame: str | None = None,
        pose_source: str | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT INTO cameras (camera_id, location_label, source_type, pose_frame, pose_source) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(camera_id) DO UPDATE SET "
            "  location_label = excluded.location_label,"
            "  pose_frame = excluded.pose_frame,"
            "  pose_source = excluded.pose_source",
            (camera_id, location_label, source_type, pose_frame, pose_source),
        )
        self.connection.commit()

    def add(
        self,
        camera_id: str,
        captured_at: dt.datetime,
        source_sequence: int,
        embedding: np.ndarray,
        pose: Pose,
        detections: Sequence[Detection] = (),
        timestamp_source: str = "host_receive",
        device_timestamp_ns: int | None = None,
        frame_id: str | None = None,
    ) -> StoredFrame:
        """Write one record: embedding first, then the metadata that references it."""
        vector = np.asarray(embedding, dtype=DTYPE).reshape(-1)
        if vector.shape[0] != self.embedding_dim:
            raise ValueError(f"embedding has {vector.shape[0]} values, collection expects {self.embedding_dim}")
        captured_at = captured_at.astimezone(dt.timezone.utc)
        frame_id = frame_id or make_frame_id(camera_id, captured_at, source_sequence)
        partition = partition_for(captured_at)
        row = self._append_vector(camera_id, partition, vector)

        frame = StoredFrame(
            frame_id=frame_id,
            camera_id=camera_id,
            captured_at=captured_at,
            source_sequence=int(source_sequence),
            pose=pose,
            detections=tuple(detections),
            vector_partition=partition,
            vector_row=row,
        )
        connection = self.connection
        with connection:  # one transaction: frames row and its detections together
            connection.execute(
                "INSERT INTO frames (frame_id, camera_id, captured_at, timestamp_source, source_sequence,"
                " device_timestamp_ns, ingested_at, collection_id, vector_partition, vector_row,"
                " pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, pose_frame, pose_source,"
                " detection_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    frame_id,
                    camera_id,
                    captured_at.isoformat(timespec="milliseconds"),
                    timestamp_source,
                    int(source_sequence),
                    device_timestamp_ns,
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
                    self.collection_id,
                    partition,
                    row,
                    pose.x, pose.y, pose.z,
                    pose.qx, pose.qy, pose.qz, pose.qw,
                    pose.frame, pose.source,
                    len(detections),
                ),
            )
            connection.executemany(
                "INSERT INTO detections (frame_id, label, label_id, confidence, x1, y1, x2, y2) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (frame_id, d.label, d.label_id, d.confidence, *d.bbox_norm)
                    for d in detections
                ],
            )
        return frame

    def _append_vector(self, camera_id: str, partition: str, vector: np.ndarray) -> int:
        directory = self.vectors_root / camera_id
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / f"{partition}.f16"
        meta_path = directory / f"{partition}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
        if meta is None:
            rows = data_path.stat().st_size // (self.embedding_dim * DTYPE_SIZE) if data_path.is_file() else 0
            meta = {"dim": self.embedding_dim, "dtype": "float16", "rows": rows, "collection_id": self.collection_id}
        elif meta["dim"] != self.embedding_dim:
            raise RuntimeError(f"{meta_path} declares dim {meta['dim']}, expected {self.embedding_dim}")

        # A crash may leave complete vector bytes beyond the last sidecar update.
        # Never reuse that row number: append mode writes at EOF regardless of seek.
        byte_count = data_path.stat().st_size if data_path.is_file() else 0
        row_bytes = self.embedding_dim * DTYPE_SIZE
        if byte_count % row_bytes or byte_count // row_bytes < int(meta["rows"]):
            raise RuntimeError(f"{data_path} is truncated; refusing to append")
        row = byte_count // row_bytes
        with data_path.open("ab") as handle:
            handle.seek(row * self.embedding_dim * DTYPE_SIZE)
            handle.write(vector.astype(DTYPE).tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        meta["rows"] = row + 1
        _atomic_json(meta_path, meta)
        return row

    # -- read -------------------------------------------------------------------

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])

    def _where(
        self,
        camera_id: str | None,
        start: dt.datetime | None,
        end: dt.datetime | None,
        labels: Sequence[str] | None,
        alias: str = "",
    ) -> tuple[str, list[Any]]:
        prefix = f"{alias}." if alias else ""
        where: list[str] = []
        params: list[Any] = []
        if camera_id:
            where.append(f"{prefix}camera_id = ?")
            params.append(camera_id)
        if start:
            where.append(f"{prefix}captured_at >= ?")
            params.append(start.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds"))
        if end:
            where.append(f"{prefix}captured_at < ?")
            params.append(end.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds"))
        if labels:
            placeholders = ",".join("?" for _ in labels)
            where.append(
                f"{prefix}frame_id IN (SELECT frame_id FROM detections WHERE label IN ({placeholders}))"
            )
            params.extend(labels)
        return (" AND ".join(where) if where else "1=1"), params

    def iter_frames(
        self,
        camera_id: str | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        labels: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[StoredFrame]:
        """Yield matching frames, oldest first.

        Detections for the whole result set are fetched in one query rather than
        one per frame: at 86k frames the per-frame query cost about a second on
        its own.
        """
        where, params = self._where(camera_id, start, end, labels)
        sql = f"SELECT * FROM frames WHERE {where} ORDER BY captured_at"
        if limit:
            sql += " LIMIT ?"
        rows = list(self.connection.execute(sql, params + ([int(limit)] if limit else [])))

        detections: dict[str, list[Detection]] = {}
        if rows:
            detection_where, detection_params = self._where(camera_id, start, end, labels, alias="f")
            for row in self.connection.execute(
                "SELECT d.* FROM detections d JOIN frames f ON d.frame_id = f.frame_id "
                f"WHERE {detection_where} ORDER BY d.confidence DESC, d.id",
                detection_params,
            ):
                detections.setdefault(row["frame_id"], []).append(
                    Detection(
                        label=row["label"],
                        label_id=int(row["label_id"]),
                        confidence=float(row["confidence"]),
                        bbox_norm=(row["x1"], row["y1"], row["x2"], row["y2"]),
                    )
                )

        for row in rows:
            yield self._to_frame(row, detections.get(row["frame_id"], ()))

    def get_frame(self, frame_id: str) -> StoredFrame | None:
        row = self.connection.execute("SELECT * FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
        return self._to_frame(row) if row else None

    def _to_frame(self, row: sqlite3.Row, detections: Sequence[Detection] | None = None) -> StoredFrame:
        """`detections=None` means "look them up"; an empty sequence means "there are none"."""
        if detections is None:
            detections = [
                Detection(
                    label=d["label"],
                    label_id=int(d["label_id"]),
                    confidence=float(d["confidence"]),
                    bbox_norm=(d["x1"], d["y1"], d["x2"], d["y2"]),
                )
                for d in self.connection.execute(
                    "SELECT * FROM detections WHERE frame_id = ? ORDER BY confidence DESC, id",
                    (row["frame_id"],),
                )
            ]
        return StoredFrame(
            frame_id=row["frame_id"],
            camera_id=row["camera_id"],
            captured_at=dt.datetime.fromisoformat(row["captured_at"]),
            source_sequence=int(row["source_sequence"]),
            pose=Pose(
                x=row["pose_x"], y=row["pose_y"], z=row["pose_z"],
                qx=row["pose_qx"], qy=row["pose_qy"], qz=row["pose_qz"], qw=row["pose_qw"],
                frame=row["pose_frame"], source=row["pose_source"],
            ),
            detections=tuple(detections),
            vector_partition=row["vector_partition"],
            vector_row=int(row["vector_row"]),
            device_timestamp_ns=(
                int(row["device_timestamp_ns"]) if row["device_timestamp_ns"] is not None else None
            ),
        )

    def partition_path(self, camera_id: str, partition: str) -> Path:
        return self.vectors_root / camera_id / f"{partition}.f16"

    def open_partition(self, camera_id: str, partition: str) -> np.memmap:
        """Memory-map one partition for reading, in its stored float16 dtype.

        Callers gather rows by index; nothing here copies until asked.
        """
        path = self.partition_path(camera_id, partition)
        meta_path = path.with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} is missing; the partition cannot be sized (run verify())")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["dim"] != self.embedding_dim:
            raise RuntimeError(f"{meta_path} declares dim {meta['dim']}, expected {self.embedding_dim}")
        rows = int(meta["rows"])
        if rows == 0:
            return np.empty((0, self.embedding_dim), dtype=DTYPE)
        return np.memmap(path, dtype=DTYPE, mode="r", shape=(rows, self.embedding_dim))

    def load_vectors(self, frames: Iterable[StoredFrame]) -> np.ndarray:
        """Read back embeddings as float32 `[N, D]`, in the order given.

        One memory-map per partition, not one file open per row: reopening per
        row cost ~98 us each on this machine, which is ~10 s for 100k frames and
        blows the query budget in spec section 12.

        Re-normalised after the float16 round trip, as spec section 6 requires,
        so a plain dot product against a unit-norm query is a cosine.
        """
        frames = list(frames)
        if not frames:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for position, frame in enumerate(frames):
            grouped.setdefault((frame.camera_id, frame.vector_partition), []).append(
                (position, frame.vector_row)
            )

        matrix = np.empty((len(frames), self.embedding_dim), dtype=np.float32)
        for (camera_id, partition), items in grouped.items():
            mapped = self.open_partition(camera_id, partition)
            rows = [row for _, row in items]
            if rows and (max(rows) >= mapped.shape[0] or min(rows) < 0):
                raise RuntimeError(
                    f"{self.partition_path(camera_id, partition)} has {mapped.shape[0]} rows but row "
                    f"{max(rows)} is referenced; the vector store and the database disagree (run verify())"
                )
            matrix[[position for position, _ in items]] = np.asarray(mapped[rows], dtype=np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("a stored embedding has zero norm; the vector store is corrupt")
        return matrix / norms

    def verify(self) -> dict:
        """Cross-check the database against the vector store."""
        problems: list[str] = []
        partitions: dict[tuple[str, str], int] = {}
        for directory in self.vectors_root.iterdir() if self.vectors_root.is_dir() else []:
            if not directory.is_dir():
                continue
            for meta_path in directory.glob("*.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    partitions[(directory.name, meta_path.stem)] = int(meta["rows"])
                except (OSError, ValueError, TypeError, KeyError) as error:
                    problems.append(f"partition {directory.name}/{meta_path.stem} has an invalid sidecar: {error}")

        referenced: dict[tuple[str, str], set[int]] = {}
        for row in self.connection.execute("SELECT camera_id, vector_partition, vector_row, frame_id FROM frames"):
            key = (row["camera_id"], row["vector_partition"])
            referenced.setdefault(key, set()).add(int(row["vector_row"]))

        for key, rows in referenced.items():
            declared = partitions.get(key)
            if declared is None:
                problems.append(f"partition {key[0]}/{key[1]} is referenced but has no sidecar")
                continue
            if max(rows) >= declared:
                problems.append(f"partition {key[0]}/{key[1]} declares {declared} rows but row {max(rows)} is referenced")
        for key, declared in partitions.items():
            used = len(referenced.get(key, ()))
            if used != declared:
                problems.append(f"partition {key[0]}/{key[1]}: {declared} rows on disk, {used} referenced by the database")

        return {
            "collection_id": self.collection_id,
            "frames": self.count(),
            "partitions": len(partitions),
            "orphan_or_missing": problems,
            "ok": not problems,
        }

    def time_range(self, camera_id: str | None = None) -> tuple[dt.datetime | None, dt.datetime | None]:
        """Earliest and latest `captured_at` in the collection, ignoring any window."""
        sql = "SELECT MIN(captured_at) AS first, MAX(captured_at) AS last FROM frames"
        params: list[Any] = []
        if camera_id:
            sql += " WHERE camera_id = ?"
            params.append(camera_id)
        row = self.connection.execute(sql, params).fetchone()
        if row is None or row["first"] is None:
            return None, None
        return dt.datetime.fromisoformat(row["first"]), dt.datetime.fromisoformat(row["last"])

    def camera_locations(self) -> dict[str, str | None]:
        """Human-readable place per camera.

        The only honest answer to "where" while poses are simulated: a place the
        operator named, rather than a coordinate that came out of a stand-in
        trajectory.
        """
        return {
            row["camera_id"]: row["location_label"]
            for row in self.connection.execute("SELECT camera_id, location_label FROM cameras")
        }

    def stats(self) -> dict:
        labels = {
            row["label"]: int(row["n"])
            for row in self.connection.execute(
                "SELECT label, COUNT(*) AS n FROM detections GROUP BY label ORDER BY n DESC"
            )
        }
        return {
            "collection_id": self.collection_id,
            "model_fingerprint": self.model_fingerprint,
            "embedding_dim": self.embedding_dim,
            "frames": self.count(),
            "cameras": [
                row["camera_id"]
                for row in self.connection.execute("SELECT camera_id FROM cameras ORDER BY camera_id")
            ],
            "detection_labels": labels,
        }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
