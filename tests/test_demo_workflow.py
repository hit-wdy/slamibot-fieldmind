"""Protect pause timing, coordinate scope and count selection used in the recording."""

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("demo_support", SCRIPTS / "demo_support.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_waiting_for_user_and_pause_do_not_consume_bag():
    now = [0.]
    c = demo.ReplayControl(clock=lambda: now[0])
    now[0] = 25
    c.ready()
    now[0] = 600
    assert c.snapshot()["playback_elapsed_s"] == 0
    c.command("start-navigation")
    now[0] = 603
    assert c.snapshot()["playback_elapsed_s"] == 3
    c.command("pause")
    now[0] = 1200
    assert c.snapshot()["playback_elapsed_s"] == 3
    c.command("resume")
    now[0] = 1202
    assert c.snapshot()["playback_elapsed_s"] == 5


def test_no_start_before_ready_or_restart_after_finish():
    c = demo.ReplayControl()
    with pytest.raises(ValueError):
        c.command("start-navigation")
    c.ready()
    with pytest.raises(ValueError):
        c.command("resume")
    c.command("start-navigation")
    with pytest.raises(ValueError):
        c.command("start-navigation")
    c.finish()
    with pytest.raises(ValueError):
        c.command("resume")
    assert c.snapshot()["state"] == "complete"
    c.command("stop")
    assert c.snapshot()["state"] == "stopped"


def test_playback_speed_and_bad_speed():
    now = [0.]
    c = demo.ReplayControl(speed=2, clock=lambda: now[0])
    c.ready()
    c.command("start-navigation")
    now[0] = 1
    assert c.snapshot()["playback_elapsed_s"] == 2
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            demo.ReplayControl(bad)


def test_failed_and_stopped_do_not_keep_running_clock():
    now = [0.]
    c = demo.ReplayControl(clock=lambda: now[0])
    c.ready()
    c.command("start-navigation")
    now[0] = 5
    c.finish("failed")
    now[0] = 20
    assert c.snapshot()["playback_elapsed_s"] == 5


def test_indexing_drain_is_not_navigation_time():
    now = [0.]
    c = demo.ReplayControl(clock=lambda: now[0])
    c.ready()
    c.command("start-navigation")
    now[0] = 12
    c.drain()
    now[0] = 20
    assert c.snapshot()["state"] == "indexing"
    assert c.snapshot()["playback_elapsed_s"] == 12
    with pytest.raises(ValueError):
        c.command("resume")
    c.finish()
    assert c.snapshot()["state"] == "complete"


def test_place_registration_uses_backend_put_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("demo_replay", SCRIPTS / "demo_replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    seen = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"status":"ok"}'
    def open_url(request, **kwargs):
        seen.append(request)
        return Response()
    monkeypatch.setattr(replay.urllib.request, "urlopen", open_url)
    assert replay.http_json("http://backend/v1/places", {"name": "p1"}, method="PUT") == {"status": "ok"}
    assert seen[0].get_method() == "PUT"


def test_navigation_points_match_synthetic_trajectory_and_scope():
    points = demo.navigation_points("coord1", "boot1", 235)
    assert [p["x"] for p in points] == [5, 20, 35, 50]
    assert len({p["place_id"] for p in points}) == 4
    frame = dict(coordinate_id="coord1", camera_pose=demo.simulated_pose(140))
    result = demo.relative_place(frame, points)
    assert result["place_id"].endswith("position-3")
    assert result["camera_distance_m"] == 0
    assert result["pose_source"] == "simulated"
    assert demo.relative_place(dict(frame, coordinate_id="another-map"), points) is None
    assert demo.relative_place(dict(frame, camera_pose=None), points) is None
    assert demo.navigation_points("coord1", "boot1", 20) == []


def frame(count, available=True):
    return dict(frame_id="f" + str(count), detection_status="available" if available else "unavailable",
                detections=[dict(label="chair") for _ in range(count)],
                label_counts={"chair": 999})  # Never trust stale/incorrect derived counts.


def test_exact_count_filter_keeps_original_ranks_and_does_not_use_raw_detections():
    rows = [frame(1), frame(4), frame(4, False), frame(5), frame(4)]
    rows[0]["detections_raw"] = [dict(label="chair")] * 4
    selected = demo.choose_candidates(rows, "chair", 4)
    assert [r["radio_rank"] for r in selected] == [2, 5]
    assert all("radio_rank" not in r for r in rows)
    assert demo.class_count(frame(4, False), "chair") is None


def test_plant_vase_geometry_is_not_color_verification(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("demo_mcp_query", SCRIPTS / "demo_mcp_query.py")
    query = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(query)
    row = dict(detections=[dict(label="potted plant", bbox_norm=[.1, .1, .5, .9]),
                           dict(label="vase", bbox_norm=[.2, .5, .4, .85]),
                           dict(label="vase", bbox_norm=[.7, .5, .9, .8])])
    pairs = query.pot_pairs(row)
    assert len(pairs) == 1
    assert "color" not in pairs[0]
