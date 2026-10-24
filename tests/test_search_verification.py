"""RADIO-first retrieval with per-frame YOLO evidence, without model downloads."""

import datetime as dt
import json
import sys

import anyio
import numpy as np
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from slamibot_fieldmind.detect import Detection
from slamibot_fieldmind.pose import SimulatedPoseSource
from slamibot_fieldmind.query import QueryService
from slamibot_fieldmind.store import RecordStore


class FixedEncoder:
    def encode_texts(self, texts):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


@pytest.fixture
def recorded(tmp_path):
    start = dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc)
    pose = SimulatedPoseSource()
    ids = []
    with RecordStore(tmp_path, "test", "f" * 64, 2) as store:
        store.register_camera("camera_01", location_label="room")
        # The strongest semantic match deliberately has no person detection.
        for i, (vector, detections) in enumerate([
            ([1.0, 0.0], []),
            ([0.8, 0.6], [Detection("person", 0, 0.9, (0.1, 0.2, 0.5, 0.9))]),
            ([0.0, 1.0], [Detection("truck", 7, 0.95, (0.2, 0.2, 0.8, 0.8))]),
        ]):
            ids.append(store.add(
                "camera_01", start + dt.timedelta(seconds=i), i,
                np.array(vector, dtype=np.float32), pose.at(i), detections,
            ).frame_id)
    window = {"start_time": start.isoformat(), "end_time": (start + dt.timedelta(seconds=3)).isoformat()}
    return tmp_path, QueryService(tmp_path, "test", encoder_factory=FixedEncoder), ids, window


@pytest.mark.parametrize("group_results", [False, True])
def test_yolo_evidence_does_not_hide_or_reorder_semantic_candidates(recorded, group_results):
    _, service, ids, window = recorded
    result = service.search("a person", top_k=3, group_results=group_results, group_window_s=0, **window)
    assert result["retrieval_strategy"] == "radio_then_yolo"
    assert [f["frame_id"] for f in result["matches"]] == ids
    assert result["matches"][0]["yolo_verification"]["status"] == "not_detected"
    evidence = result["matches"][1]["yolo_verification"]
    assert evidence["status"] == "supported_by_yolo"
    assert evidence["detections"][0]["bbox_norm"] == [0.1, 0.2, 0.5, 0.9]
    assert all(d["label"] == "person" for d in evidence["detections"])


@pytest.mark.parametrize("query", ["a ceiling light", "a large whale", "a fire truck"])
def test_unsupported_targets_return_the_radio_best_match(recorded, query):
    _, service, ids, window = recorded
    result = service.search(query, top_k=1, **window)
    assert result["retrieval_strategy"] == "radio_only"
    assert result["matches"][0]["frame_id"] == ids[0]
    assert result["matches"][0]["yolo_verification"]["status"] == "not_supported"


def test_descriptive_query_verifies_only_the_explicit_object_class(recorded):
    _, service, _, window = recorded
    result = service.search("a person wearing glasses beside a truck", target_label="person", group_results=False, **window)
    assert result["verification_target"] == "person"
    assert result["matches"][1]["yolo_verification"]["status"] == "supported_by_yolo"
    assert result["matches"][2]["yolo_verification"]["status"] == "not_detected"


def test_supported_class_without_detections_still_returns_candidate(recorded):
    _, service, ids, window = recorded
    result = service.search("a bed", top_k=1, **window)
    assert result["retrieval_strategy"] == "radio_then_yolo"
    assert result["matches"][0]["frame_id"] == ids[0]
    assert result["matches"][0]["yolo_verification"]["status"] == "not_detected"


def test_bad_old_sidecar_is_reported_without_blocking_status_or_intact_queries(recorded):
    root, service, ids, window = recorded
    old = root / "vectors" / "test" / "camera_01" / "20261014T16.json"
    old.write_bytes(b"\x00" * 79)
    status = service.status()
    assert status["vector_partitions_ok"] is False
    assert any("20261014T16" in problem and "invalid sidecar" in problem for problem in status["integrity_problems"])
    result = service.search("a person", top_k=1, **window)
    assert result["matches"][0]["frame_id"] == ids[0]


def test_mcp_exposes_and_forwards_the_verification_target(recorded):
    root, _, ids, window = recorded
    code = (
        "import sys, numpy as np; "
        "from slamibot_fieldmind.query import QueryService; "
        "from slamibot_fieldmind.mcp_server import build_server; "
        "from types import SimpleNamespace; "
        "factory=lambda: SimpleNamespace(encode_texts=lambda texts: np.array([[1.,0.]]*len(texts),dtype=np.float32)); "
        "build_server(QueryService(sys.argv[1],'test',encoder_factory=factory)).run(transport='stdio')"
    )

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-c", code, str(root)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                search = next(t for t in tools if t.name == "search_frames")
                assert "target_label" in search.input_schema["properties"]
                for target, expected in [("person", "radio_then_yolo"), ("ceiling light", "radio_only")]:
                    result = await session.call_tool("search_frames", {
                        "query": "a photo of the target", "target_label": target,
                        "top_k": 1, **window,
                    })
                    assert not result.is_error
                    payload = json.loads(result.content[0].text)
                    assert payload["retrieval_strategy"] == expected
                    assert payload["matches"][0]["frame_id"] == ids[0]

    anyio.run(run)
