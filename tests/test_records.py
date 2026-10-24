"""Pose, sampling and the record store.

These run without the camera and without the model, so they cover the parts of
the record path that a flaky USB link cannot take away.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pytest

from slamibot_fieldmind.detect import Detection
from slamibot_fieldmind.pose import SIMULATED, Pose, SimulatedPoseSource
from slamibot_fieldmind.sampling import IntervalSampler
from slamibot_fieldmind.store import SCHEMA_VERSION, RecordStore, make_frame_id, partition_for

DIM = 8
FINGERPRINT = "c" * 64


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


# -- pose -----------------------------------------------------------------------


def test_simulated_pose_is_deterministic():
    source = SimulatedPoseSource(seed=3)
    assert source.at(12.5) == source.at(12.5)
    assert source.at(12.5) != SimulatedPoseSource(seed=4).at(12.5)


def test_simulated_pose_is_labelled_as_simulated():
    """The one property that must never be lost: this is not a measured pose."""
    assert SimulatedPoseSource().at(0.0).source == SIMULATED
    assert json.loads(SimulatedPoseSource().at(0.0).to_json())["source"] == SIMULATED


def test_pose_distance_matches_the_trajectory():
    source = SimulatedPoseSource(radius_m=2.0, period_s=60.0)
    half_lap = source.at(30.0)
    assert source.at(0.0).distance_to(half_lap) == pytest.approx(4.0, abs=1e-6)


def test_pose_yaw_wraps_short_way_round():
    source = SimulatedPoseSource(period_s=60.0)
    change = source.at(0.0).yaw_change_to(source.at(30.0))
    assert abs(change) == pytest.approx(180.0, abs=1e-6)
    for a, b in [(0.0, 5.0), (0.0, 55.0), (10.0, 20.0)]:
        assert -180.0 <= source.at(a).yaw_change_to(source.at(b)) <= 180.0


def test_poses_in_different_frames_are_not_comparable():
    a = SimulatedPoseSource(frame="map").at(0.0)
    b = SimulatedPoseSource(frame="odom").at(0.0)
    with pytest.raises(ValueError, match="different frames"):
        a.distance_to(b)


def test_pose_json_round_trips():
    pose = SimulatedPoseSource().at(7.0)
    payload = json.loads(pose.to_json())
    assert set(payload) == {"x", "y", "z", "qx", "qy", "qz", "qw", "frame", "source"}
    assert payload["x"] == pytest.approx(pose.x, abs=1e-4)


def test_quaternion_is_normalised():
    for t in (0.0, 13.0, 41.5):
        pose = SimulatedPoseSource().at(t)
        norm = math.sqrt(pose.qx**2 + pose.qy**2 + pose.qz**2 + pose.qw**2)
        assert norm == pytest.approx(1.0, abs=1e-9)


# -- sampling -------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_sampler_keeps_one_frame_per_interval():
    clock = FakeClock()
    sampler = IntervalSampler(interval_s=1.0, clock=clock)
    kept = 0
    for _ in range(100):  # 100 frames at 10 Hz = 10 seconds
        if sampler.observe():
            kept += 1
        clock.advance(0.1)
    assert 10 <= kept <= 12, kept
    assert sampler.stats.seen == 100
    assert sampler.stats.skipped_by_plan == 100 - kept
    assert sampler.stats.observed_hz == pytest.approx(1.0, abs=0.15)


def test_sampler_is_clock_based_not_frame_counted():
    """Dropped frames must not stretch the interval: with 5 Hz input and a 1 s
    interval, every fifth frame arriving would still give 1 Hz in wall time,
    whereas a frame counter tied to 10 Hz input would halve it."""
    clock = FakeClock()
    sampler = IntervalSampler(interval_s=1.0, clock=clock)
    kept = 0
    for _ in range(50):  # 50 frames at 5 Hz = 10 seconds
        if sampler.observe():
            kept += 1
        clock.advance(0.2)
    assert 9 <= kept <= 11, kept


def test_sampler_counts_falling_behind():
    clock = FakeClock()
    sampler = IntervalSampler(interval_s=1.0, clock=clock)
    sampler.observe()
    clock.advance(5.0)  # the loop stalled for five intervals
    sampler.observe()
    assert sampler.stats.overdue_recoveries == 1
    assert sampler.stats.longest_gap_s == pytest.approx(5.0, abs=1e-6)


def test_sampler_rejects_a_non_positive_interval():
    with pytest.raises(ValueError):
        IntervalSampler(interval_s=0.0)


# -- store ----------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    with RecordStore(tmp_path, "c1", FINGERPRINT, DIM, {"index_interval_s": 1.0}) as opened:
        opened.register_camera("camera_01", pose_frame="map", pose_source=SIMULATED)
        yield opened


def test_frame_id_is_not_an_array_index():
    moment = dt.datetime(2026, 10, 11, 12, 0, 0, 123000, tzinfo=dt.timezone.utc)
    frame_id = make_frame_id("camera_01", moment, 42)
    assert frame_id == "camera_01-20261011T120000.123-000042"


def test_partition_is_hourly_in_utc():
    moment = dt.datetime(2026, 10, 11, 23, 59, 59, tzinfo=dt.timezone.utc)
    assert partition_for(moment) == "20261011T23"
    plus_offset = dt.datetime(2026, 10, 12, 7, 59, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    assert partition_for(plus_offset) == "20261011T23", "the same instant must land in the same partition"


def test_round_trip_preserves_vectors_and_metadata(store):
    source = SimulatedPoseSource()
    base = dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    written = []
    for index in range(4):
        written.append(
            store.add(
                camera_id="camera_01",
                captured_at=base + dt.timedelta(seconds=index),
                source_sequence=100 + index,
                embedding=unit(index),
                pose=source.at(index),
                detections=[Detection("person", 0, 0.9 - 0.1 * index, (0.1, 0.1, 0.2, 0.2))],
            )
        )
    frames = list(store.iter_frames())
    assert len(frames) == 4
    assert [f.frame_id for f in frames] == [w.frame_id for w in written]

    back = store.load_vectors(frames)
    for index, frame in enumerate(frames):
        assert float(back[index] @ unit(index)) == pytest.approx(1.0, abs=1e-4)
        assert np.linalg.norm(back[index]) == pytest.approx(1.0, abs=1e-6)
        assert frame.pose == source.at(index)
        assert frame.detections[0].label == "person"
        assert frame.label_counts == {"person": 1}


def test_vectors_are_reloaded_unit_norm(store):
    """Spec section 6: float16 on disk, float32 and re-normalised on load."""
    store.add("camera_01", dt.datetime.now(dt.timezone.utc), 1, unit(1) * 0.9, SimulatedPoseSource().at(0.0))
    loaded = store.load_vectors(list(store.iter_frames()))
    assert np.linalg.norm(loaded[0]) == pytest.approx(1.0, abs=1e-6)


def test_label_and_time_filters(store):
    base = dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    store.add("camera_01", base, 1, unit(1), SimulatedPoseSource().at(0.0), [Detection("truck", 7, 0.8, (0, 0, 1, 1))])
    store.add("camera_01", base + dt.timedelta(seconds=1), 2, unit(2), SimulatedPoseSource().at(1.0), [])
    store.add("camera_01", base + dt.timedelta(seconds=2), 3, unit(3), SimulatedPoseSource().at(2.0), [Detection("truck", 7, 0.5, (0, 0, 1, 1))])

    assert len(list(store.iter_frames(labels=["truck"]))) == 2
    assert len(list(store.iter_frames(labels=["person"]))) == 0
    window = list(store.iter_frames(start=base, end=base + dt.timedelta(seconds=1)))
    assert [f.source_sequence for f in window] == [1]
    assert len(list(store.iter_frames(limit=2))) == 2


def test_camera_filter_isolates_cameras(store):
    base = dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    store.add("camera_01", base, 1, unit(1), SimulatedPoseSource().at(0.0))
    store.add("camera_02", base, 2, unit(2), SimulatedPoseSource().at(0.0))
    assert [f.camera_id for f in store.iter_frames(camera_id="camera_02")] == ["camera_02"]


def test_wrong_embedding_dimension_is_refused(store):
    with pytest.raises(ValueError, match="collection expects"):
        store.add("camera_01", dt.datetime.now(dt.timezone.utc), 1, unit(1, dim=DIM + 1), SimulatedPoseSource().at(0.0))


def test_verify_is_clean_after_a_normal_run(store):
    base = dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    for index in range(3):
        store.add("camera_01", base + dt.timedelta(hours=index), index, unit(index), SimulatedPoseSource().at(index))
    report = store.verify()
    assert report["ok"], report["orphan_or_missing"]
    assert report["partitions"] == 3


def test_verify_notices_a_partition_that_lost_rows(store, tmp_path):
    store.add("camera_01", dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc), 1, unit(1), SimulatedPoseSource().at(0.0))
    store.add("camera_01", dt.datetime(2026, 10, 11, 12, 0, 1, tzinfo=dt.timezone.utc), 2, unit(2), SimulatedPoseSource().at(1.0))
    meta_path = next((tmp_path / "vectors" / "c1" / "camera_01").glob("*.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["rows"] = 0
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    report = store.verify()
    assert not report["ok"]
    assert any("declares 0 rows" in problem for problem in report["orphan_or_missing"])


def test_a_collection_refuses_a_different_fingerprint(tmp_path):
    """Spec 2.3: vectors from different fingerprints must never be mixed."""
    with RecordStore(tmp_path, "c1", FINGERPRINT, DIM) as store:
        store.add("camera_01", dt.datetime.now(dt.timezone.utc), 1, unit(1), SimulatedPoseSource().at(0.0))
    with pytest.raises(RuntimeError, match="different space"):
        RecordStore(tmp_path, "c1", "d" * 64, DIM).open()


def test_a_collection_refuses_a_different_dimension(tmp_path):
    with RecordStore(tmp_path, "c1", FINGERPRINT, DIM) as store:
        store.add("camera_01", dt.datetime.now(dt.timezone.utc), 1, unit(1), SimulatedPoseSource().at(0.0))
    with pytest.raises(RuntimeError, match="has dim"):
        RecordStore(tmp_path, "c1", FINGERPRINT, DIM * 2).open()


def test_no_images_are_written(store, tmp_path):
    """The requirement is explicit: records only, no frames on disk."""
    store.add(
        "camera_01",
        dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc),
        1,
        unit(1),
        SimulatedPoseSource().at(0.0),
        [Detection("truck", 7, 0.9, (0, 0, 1, 1))],
    )
    suffixes = {path.suffix for path in tmp_path.rglob("*") if path.is_file()}
    allowed = {".sqlite", ".sqlite-wal", ".sqlite-shm", ".f16", ".json", ".tmp"}
    assert suffixes <= allowed, f"unexpected files written: {suffixes - allowed}"


def test_an_old_schema_is_refused_not_silently_upgraded(tmp_path):
    """A pre-versioning database must fail loudly: CREATE TABLE IF NOT EXISTS
    would leave the old nullable columns in place and the code would trip over
    them later, far from the cause."""
    import sqlite3

    old = sqlite3.connect(tmp_path / "records.sqlite")
    old.executescript(
        "CREATE TABLE frames (frame_id TEXT PRIMARY KEY, camera_id TEXT, captured_at TEXT,"
        " timestamp_source TEXT, source_sequence INTEGER, ingested_at TEXT, collection_id TEXT,"
        " vector_partition TEXT, vector_row INTEGER,"
        " pose_x REAL, pose_y REAL, pose_z REAL,"
        " pose_qx REAL, pose_qy REAL, pose_qz REAL, pose_qw REAL,"
        " pose_frame TEXT NOT NULL, pose_source TEXT NOT NULL, detection_count INTEGER NOT NULL DEFAULT 0)"
    )
    old.commit()
    old.close()

    with pytest.raises(RuntimeError, match="schema version 0"):
        RecordStore(tmp_path, "c1", FINGERPRINT, DIM).open()


def test_a_fresh_database_is_accepted(tmp_path):
    with RecordStore(tmp_path, "c1", FINGERPRINT, DIM) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_the_device_clock_is_kept_for_slam_correlation(store):
    """A SLAM front end keys off the monotonic device clock, not the wall clock.

    Dropping this would leave no way to line a pose up with the frame it belongs
    to, so the stored pose could only ever be guessed at.
    """
    frame = store.add(
        "camera_01",
        dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc),
        11,
        unit(1),
        SimulatedPoseSource().at(0.0),
        device_timestamp_ns=8_020_704_000,
    )
    assert store.get_frame(frame.frame_id).device_timestamp_ns == 8_020_704_000


def test_a_missing_device_clock_is_allowed(store):
    frame = store.add("camera_01", dt.datetime.now(dt.timezone.utc), 1, unit(1), SimulatedPoseSource().at(0.0))
    assert store.get_frame(frame.frame_id).device_timestamp_ns is None


def test_frames_cannot_be_stored_without_a_pose(store):
    """Every record carries a pose; `pose_source` says whether it is measured."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
        store.connection.execute(
            "INSERT INTO frames (frame_id, camera_id, captured_at, timestamp_source, source_sequence,"
            " ingested_at, collection_id, vector_partition, vector_row, pose_frame, pose_source,"
            " detection_count) VALUES ('f','camera_01','t','host_receive',1,'n','c1','p',0,'map','simulated',0)"
        )


def test_stats_summarises_labels(store):
    base = dt.datetime(2026, 10, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    store.add("camera_01", base, 1, unit(1), SimulatedPoseSource().at(0.0), [Detection("truck", 7, 0.9, (0, 0, 1, 1))])
    store.add("camera_01", base + dt.timedelta(seconds=1), 2, unit(2), SimulatedPoseSource().at(1.0), [Detection("person", 0, 0.8, (0, 0, 1, 1))])
    stats = store.stats()
    assert stats["frames"] == 2
    assert stats["detection_labels"] == {"person": 1, "truck": 1}
    assert stats["cameras"] == ["camera_01"]
