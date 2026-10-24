"""Offline refresh is transactional, backed up and excluded by the backend lock."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest
from PIL import Image

pytest.importorskip("openvino")
pytest.importorskip("cv2")
scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("refine_detections", scripts / "refine_detections.py")
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


@pytest.fixture
def sample(tmp_path, monkeypatch):
    root = tmp_path / "index"
    (root / "images").mkdir(parents=True)
    (root / "vectors").mkdir()
    (root / "vectors" / "unchanged.f16").write_bytes(b"VECTOR-EVIDENCE")
    Image.new("RGB", (64, 48), "white").save(root / "images/frame.jpg")
    old = dict(label="bed", label_id=59, confidence=0.9, bbox_norm=[0.1, 0.1, 0.8, 0.8])
    new = dict(label="chair", label_id=56, confidence=0.95, bbox_norm=[0.2, 0.2, 0.5, 0.7])
    with sqlite3.connect(root / "records.sqlite") as db:
        db.executescript("""CREATE TABLE observations(frame_id TEXT, session_id TEXT, image_path TEXT,
            radio_status TEXT, source_stamp_ns INTEGER, metadata_json TEXT);
            CREATE TABLE frames(frame_id TEXT PRIMARY KEY, detection_count INTEGER);
            CREATE TABLE detections(frame_id TEXT,label TEXT,label_id INTEGER,confidence REAL,x1 REAL,y1 REAL,x2 REAL,y2 REAL);""")
        db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?)", ("frame", "session", "images/frame.jpg", "ready", 1, json.dumps(dict(detections=[old], camera_pose={"source": "simulated"}))))
        db.execute("INSERT INTO frames VALUES ('frame',1)")
        db.execute("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?)", ("frame", old["label"], 59, 0.9, *old["bbox_norm"]))

    class Detector:
        fingerprint, parameters, flip_checks, total_ms = "new-model", {}, 0, 1.0

        def __init__(self, *args, **kwargs):
            pass

        def detect_observation(self, jpeg):
            return [new], 1.0, dict(detections_raw=[new], detection_postprocess=dict(rejected=[]))

    monkeypatch.setattr(refine, "ReplayYolo", Detector)
    return root, old, new


def invoke(monkeypatch, root, out, apply=False):
    args = ["refine", "--root", str(root), "--model", "unused.xml", "--out", str(out), "--session-id", "session"]
    if apply:
        args.append("--apply")
    monkeypatch.setattr(sys, "argv", args)
    refine.main()


def test_comparison_does_not_modify_database(sample, tmp_path, monkeypatch):
    root, old, _ = sample
    invoke(monkeypatch, root, tmp_path / "comparison")
    with sqlite3.connect(root / "records.sqlite") as db:
        assert json.loads(db.execute("SELECT metadata_json FROM observations").fetchone()[0])["detections"] == [old]


def test_apply_updates_both_indexes_and_preserves_backup_and_vectors(sample, tmp_path, monkeypatch):
    root, old, new = sample
    out = tmp_path / "applied"
    invoke(monkeypatch, root, out, True)
    with sqlite3.connect(root / "records.sqlite") as db:
        metadata = json.loads(db.execute("SELECT metadata_json FROM observations").fetchone()[0])
        assert metadata["detections"] == [new] and metadata["detections_original"] == [old]
        assert db.execute("SELECT label FROM detections").fetchone()[0] == "chair"
        assert db.execute("SELECT detection_count FROM frames").fetchone()[0] == 1
    with sqlite3.connect(out / "records.before.sqlite") as db:
        assert db.execute("SELECT label FROM detections").fetchone()[0] == "bed"
    assert (root / "vectors/unchanged.f16").read_bytes() == b"VECTOR-EVIDENCE"
    second = tmp_path / "second-pass"
    invoke(monkeypatch, root, second, True)
    with sqlite3.connect(root / "records.sqlite") as db:
        metadata = json.loads(db.execute("SELECT metadata_json FROM observations").fetchone()[0])
        assert metadata["detections_original"] == [old]
    assert json.loads((second / "summary.json").read_text())["baseline_counts"] == {"bed": 1}


def test_apply_refuses_live_backend_owner(sample, tmp_path, monkeypatch):
    root, _, _ = sample
    owner = refine.BackendLock(root)
    try:
        with pytest.raises(RuntimeError, match="another backend owns"):
            invoke(monkeypatch, root, tmp_path / "blocked", True)
    finally:
        owner.close()


def test_failed_legacy_update_rolls_back_observation(sample, tmp_path, monkeypatch):
    root, old, _ = sample
    with sqlite3.connect(root / "records.sqlite") as db:
        db.execute("DELETE FROM frames")
    with pytest.raises(RuntimeError, match="no vector record"):
        invoke(monkeypatch, root, tmp_path / "failed", True)
    with sqlite3.connect(root / "records.sqlite") as db:
        assert json.loads(db.execute("SELECT metadata_json FROM observations").fetchone()[0])["detections"] == [old]
