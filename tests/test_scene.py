"""Scene inventory must distinguish live capture, old observations and detector failure."""

import datetime as dt
import json
import sys

import anyio
import numpy as np
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from slamibot_fieldmind.capture import Frame
from slamibot_fieldmind.detect import Detection
from slamibot_fieldmind.pose import SimulatedPoseSource
from slamibot_fieldmind.scene import SceneService
from slamibot_fieldmind.store import RecordStore


PERSON = Detection("person", 0, 0.9, (0.1, 0.2, 0.5, 0.9))


class FakeCamera:
    settle_info = {"settled": True}

    def __init__(self, available=True, detections=(PERSON,), fail=False):
        self.available = available
        self.detections = detections
        self.fail = fail
        self.closed = False

    def open(self):
        if self.fail:
            raise RuntimeError("device held by recorder")

    def flush(self):
        return 0

    def grab(self):
        return Frame(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            captured_at=dt.datetime.now(dt.timezone.utc), timestamp_source="host_receive",
            source_sequence=42, camera_id="camera_01", detections=self.detections,
            detections_available=self.available,
        )

    def close(self):
        self.closed = True


def test_current_view_works_without_record_store_or_radio(tmp_path):
    camera = FakeCamera()
    service = SceneService(tmp_path, "missing", source_factory=lambda **kwargs: camera)
    result = service.describe()
    assert result["status"] == "ok"
    assert result["radio_used"] is False
    assert result["source"] == "live_camera"
    assert result["views"][0]["label_counts"] == {"person": 1}
    assert result["views"][0]["location"] is None
    assert camera.closed
    assert not list(tmp_path.iterdir())


def test_missing_detector_result_is_not_reported_as_empty_scene(tmp_path):
    camera = FakeCamera(available=False, detections=())
    result = SceneService(tmp_path, "missing", lambda **kw: camera).describe()
    assert result["status"] == "detection_unavailable"
    assert result["views"][0]["label_counts"] is None
    assert result["views"][0]["detections"] is None
    assert camera.closed


def test_valid_empty_detections_are_distinct_from_detector_failure(tmp_path):
    result = SceneService(tmp_path, "missing", lambda **kw: FakeCamera(detections=())).describe()
    assert result["status"] == "ok"
    assert result["views"][0]["label_counts"] == {}
    assert "not proof" in result["caveat"]


def test_live_camera_failure_does_not_fall_back_to_old_records_and_releases_lock(tmp_path):
    camera = FakeCamera(fail=True)
    service = SceneService(tmp_path, "missing", lambda **kw: camera)
    for _ in range(2):
        result = service.describe()
        assert result["status"] == "camera_unavailable"
        assert result["views"] == []
    assert camera.closed


@pytest.fixture
def saved(tmp_path):
    now = dt.datetime.now(dt.timezone.utc)
    pose = SimulatedPoseSource()
    with RecordStore(tmp_path, "test", "f" * 64, 2) as store:
        store.register_camera("east_1", location_label="小区东门")
        store.register_camera("east_2", location_label="小区东门")
        store.register_camera("west", location_label="小区西门")
        store.register_camera("empty", location_label="车库")
        for index, (cid, seconds, detections) in enumerate([
            ("east_1", 1000, [Detection("car", 2, 0.9, (0, 0, 1, 1))]),
            ("east_1", 1, [PERSON]),
            ("east_2", 1000, [PERSON]),
            ("west", 1, [PERSON]),
        ]):
            store.add(cid, now - dt.timedelta(seconds=seconds), index,
                      np.array([1.0, 0.0], dtype=np.float32), pose.at(index), detections)
    return tmp_path, SceneService(tmp_path, "test")


def test_location_selects_latest_per_camera_and_marks_old_observations(saved):
    _, service = saved
    result = service.describe("location", location_label="小区东门", max_age_s=60)
    assert result["status"] == "stale"
    assert result["source"] == "latest_record"
    assert [v["camera_id"] for v in result["views"]] == ["east_1", "east_2"]
    assert result["views"][0]["label_counts"] == {"person": 1}, "old car detection must not be merged"
    assert not result["views"][0]["stale"]
    assert result["views"][1]["stale"]


def test_navigation_resolved_camera_id_can_select_one_view(saved):
    _, service = saved
    result = service.describe("location", camera_id="east_1", max_age_s=60)
    assert result["status"] == "ok"
    assert len(result["views"]) == 1
    assert result["views"][0]["location"] == "小区东门"


def test_location_miss_and_unobserved_camera_do_not_return_other_places(saved):
    _, service = saved
    missing = service.describe("location", location_label="不存在")
    assert missing["status"] == "location_not_found"
    assert missing["views"] == []
    mismatch = service.describe("location", location_label="小区东门", camera_id="west")
    assert mismatch["views"] == []
    empty = service.describe("location", location_label="车库")
    assert empty["status"] == "no_observations"


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"mode": "location"}, {"mode": "current", "location_label": "东门"},
    {"mode": "location", "location_label": "东门", "device_id": "hardware"},
    {"max_age_s": float("nan")}, {"max_age_s": 0},
])
def test_invalid_requests_are_rejected_without_opening_camera(tmp_path, kwargs):
    with pytest.raises(ValueError):
        SceneService(tmp_path, "test").describe(**kwargs)


def test_describe_view_round_trip_through_standard_mcp_entrypoint(saved):
    root, _ = saved

    async def run():
        params = StdioServerParameters(command=sys.executable, args=[
            "-m", "slamibot_fieldmind.mcp_server", "--root", str(root), "--collection-id", "test",
        ])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("describe_view", {
                    "mode": "location", "camera_id": "east_1", "max_age_s": 60,
                })
                assert not result.is_error
                payload = json.loads(result.content[0].text)
                assert payload["status"] == "ok"
                assert payload["radio_used"] is False
                assert payload["views"][0]["label_counts"] == {"person": 1}

    anyio.run(run)
