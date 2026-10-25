import io
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from slamibot_fieldmind.backend import create_app
from slamibot_fieldmind.backend_query import BackendQuery
from slamibot_fieldmind.backend_runtime import BackendConfig, BackendRuntime
from slamibot_fieldmind.keyframes import KeyframePolicy, rotation_deg
from slamibot_fieldmind.protocol import utcnow


class Encoder:
    embedding_dim = 2
    instances = 0

    def __init__(self):
        type(self).instances += 1

    def fingerprint(self):
        return "test-space"

    def encode_images(self, images):
        return np.array([[1, 0] for _ in images], dtype=np.float32)

    def encode_texts(self, texts):
        return np.array([[1, 0] for _ in texts], dtype=np.float32)


def pose(x=0, angle=0):
    return dict(
        x=x,
        y=0,
        z=0,
        qx=0,
        qy=0,
        qz=math.sin(math.radians(angle) / 2),
        qw=math.cos(math.radians(angle) / 2),
        frame="camera_init",
        source="slam",
    )


def jpeg():
    f = io.BytesIO()
    Image.new("RGB", (64, 48), "red").save(f, "JPEG")
    return f.getvalue()


def observation(seq=1, stamp=1_000_000_000, x=0, angle=0, **updates):
    p = pose(x, angle)
    result = dict(
        camera_id="CAM_A",
        device_id="oak1",
        source_id="boot1",
        source_sequence=seq,
        source_stamp_ns=stamp,
        received_at=utcnow(),
        session_id="s1",
        coordinate_id="c1",
        body_pose=p,
        camera_pose=p,
        detection_status="available",
        detections=[
            dict(
                label="person",
                label_id=0,
                confidence=0.9,
                bbox_norm=[0.1, 0.1, 0.9, 0.9],
            )
        ],
    )
    result.update(updates)
    return result


def wait_for(fn, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if fn():
            return
        time.sleep(0.02)
    raise AssertionError("condition timed out")


@pytest.fixture
def runtime(tmp_path):
    r = BackendRuntime(
        BackendConfig(root=tmp_path, min_free_bytes=0, require_mount=False), Encoder
    )
    wait_for(lambda: r.model_state != "loading")
    assert r.model_state == "ready", r.model_error
    r.session(
        dict(session_id="s1", coordinate_id="c1", task_id="task1", state="running")
    )
    try:
        yield r
    finally:
        r.close()


def test_spatial_temporal_policy():
    policy = KeyframePolicy()
    previous = dict(source_stamp_ns=0, camera_pose=pose())
    assert policy.decide(0, pose(), None)["reasons"] == ["first"]
    assert policy.decide(499_000_000, pose(10), previous)["reasons"] == []
    assert policy.decide(500_000_000, pose(0.5), previous)["reasons"] == ["distance"]
    assert policy.decide(500_000_000, pose(angle=15), previous)["reasons"] == [
        "rotation"
    ]
    assert policy.decide(3_000_000_000, None, previous)["reasons"] == ["timeout"]
    assert rotation_deg([0, 0, 0, 1], [0, 0, 0, -1]) == 0
    # Rotation around pitch, not just planar yaw.
    assert rotation_deg([0, 0, 0, 1], [0, math.sin(0.2), 0, math.cos(0.2)]) > 20


@pytest.mark.parametrize(
    "seconds,x,angle,reasons,status",
    [
        (3600, 0, 0, [], "stationary"),
        (3600, 0.049, 1.99, [], "stationary"),
        (2.999, 0.05, 0, [], "waiting_for_threshold"),
        (3, 0.05, 0, ["timeout"], "selected"),
        (3, 0, 2, ["timeout"], "selected"),
        (0.5, 0, 15, ["rotation"], "selected"),
        (0.499, 1, 90, [], "min_interval"),
        (-1, 1, 90, [], "time_rollback"),
    ],
)
def test_static_and_slow_motion_policy(seconds, x, angle, reasons, status):
    previous = dict(source_stamp_ns=0, camera_pose=pose())
    result = KeyframePolicy().decide(int(seconds * 1e9), pose(x, angle), previous)
    assert result["reasons"] == reasons
    assert result["status"] == status


def test_body_pose_sampling_never_invents_camera_pose(runtime):
    first = runtime.ingest(observation(camera_pose=None), jpeg())
    still = runtime.ingest(observation(2, 100_000_000_000, camera_pose=None), jpeg())
    assert still["status"] == "skipped"
    assert still["selection"]["pose_source"] == "body_pose"
    # Even when extrinsics become available, use the common body reference;
    # never subtract a current camera position from an earlier body position.
    moved = runtime.ingest(observation(3, 101_000_000_000, x=0.5), jpeg())
    assert moved["status"] == "accepted"
    assert moved["selection"]["pose_source"] == "body_pose"
    assert BackendQuery(runtime).get_frame(first["frame_id"])["camera_pose"] is None
    assert runtime.health()["counters"]["overdue_gaps"] == 0


def test_static_stream_keeps_only_one_history_image_but_refreshes_live(runtime):
    runtime.ingest(observation(), jpeg())
    image = jpeg()
    for seq in range(2, 122):
        # Alternating sub-threshold pose noise, over ten minutes.
        result = runtime.ingest(
            observation(seq, seq * 5_000_000_000, x=0.01 * (seq % 2), angle=0.5), image
        )
        assert result["status"] == "skipped"
    q = BackendQuery(runtime)
    assert q.find_frames()["matched_frame_count"] == 1
    assert len(list(runtime.config.root.glob("images/**/*.jpg"))) == 1
    assert runtime.live["CAM_A"][0]["source_sequence"] == 121
    assert len(runtime.live) == 1
    assert runtime.health()["counters"]["stationary_skipped"] == 120
    # Moving 5 cm relative to the accepted KF (not the latest skipped frame)
    # must still trigger a timeout sample.
    result = runtime.ingest(observation(122, 610_000_000_000, x=0.05), image)
    assert result["status"] == "accepted"
    assert runtime.health()["counters"]["overdue_gaps"] == 0


def test_new_frame_ids_preserve_retry_identity_across_restart(tmp_path):
    import re

    config = BackendConfig(root=tmp_path, min_free_bytes=0, require_mount=False)
    r = BackendRuntime(config, Encoder)
    try:
        r.session(
            dict(session_id="s1", coordinate_id="c1", task_id="task1", state="running")
        )
        original = observation(captured_at="2026-10-25T09:08:07.123+08:00")
        first = r.ingest(original, jpeg())
        assert re.fullmatch(r"kf-[0-9a-f]{32}-20261025T010807123Z", first["frame_id"])
        # A receive-time fallback may change between retries. The first accepted
        # frame ID and evidence path must still win, including after restart.
        retry = observation(received_at="2026-10-25T03:00:00+00:00")
        again = r.ingest(retry, jpeg())
        assert again["status"] == "duplicates"
        assert again["frame_id"] == first["frame_id"]
        wait_for(
            lambda: (
                BackendQuery(r).get_frame(first["frame_id"])["radio_status"] == "ready"
            )
        )
    finally:
        r.close()
    r = BackendRuntime(config, Encoder)
    try:
        assert r.ingest(retry, jpeg())["frame_id"] == first["frame_id"]
        assert len(list(tmp_path.glob("images/**/*.jpg"))) == 1
        assert (
            BackendQuery(r).get_frame(first["frame_id"])["image_path"]
            == first["image_path"]
        )
    finally:
        r.close()


def test_schema_v1_migration_keeps_old_frame_id_and_image(tmp_path):
    import sqlite3
    from slamibot_fieldmind.protocol import validate_observation
    from slamibot_fieldmind.semantic_store import SEMANTIC_SCHEMA
    from slamibot_fieldmind.store import SCHEMA, SCHEMA_VERSION

    m = validate_observation(observation(), jpeg())
    old_id = m.pop("observation_key")
    m["frame_id"] = old_id
    image_path = f"images/old/{old_id}.jpg"
    (tmp_path / "images/old").mkdir(parents=True)
    (tmp_path / image_path).write_bytes(jpeg())
    with sqlite3.connect(tmp_path / "records.sqlite") as db:
        db.executescript(SCHEMA + SEMANTIC_SCHEMA)
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        db.execute("INSERT INTO semantic_schema VALUES (1)")
        db.execute(
            "INSERT INTO observations(frame_id,collection_id,session_id,coordinate_id,camera_id,captured_at,source_stamp_ns,image_path,radio_status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                old_id,
                "radio_clip_v1",
                "s1",
                "c1",
                "CAM_A",
                m["captured_at"],
                m["source_stamp_ns"],
                image_path,
                "pending",
                json.dumps(m),
            ),
        )
    r = BackendRuntime(
        BackendConfig(root=tmp_path, min_free_bytes=0, require_mount=False), Encoder
    )
    try:
        q = BackendQuery(r)
        wait_for(lambda: q.get_frame(old_id)["radio_status"] == "ready")
        assert q.get_frame(old_id)["image_path"] == image_path
        assert q.image(
            old_id
        )  # Old image and newly recovered vector remain accessible.
        retry = r.ingest(observation(received_at="2026-10-25T03:00:00+00:00"), jpeg())
        assert retry["status"] == "duplicates"
        assert retry["frame_id"] == old_id
        assert len(list(tmp_path.glob("images/**/*.jpg"))) == 1
    finally:
        r.close()


def test_end_to_end_metadata_image_search_and_coordinates(runtime):
    q = BackendQuery(runtime)
    first = runtime.ingest(observation(), jpeg())
    assert first["status"] == "accepted"
    assert runtime.ingest(observation(), jpeg())["status"] == "duplicates"
    assert (
        runtime.ingest(observation(2, 1_200_000_000, x=2), jpeg())["status"]
        == "skipped"
    )
    second = runtime.ingest(observation(3, 1_500_000_000, x=0.5), jpeg())
    assert second["selection"]["reasons"] == ["distance"]
    wait_for(lambda: q.get_frame(second["frame_id"])["radio_status"] == "ready")
    result = q.search("a person", group_results=False)
    assert len(result["matches"]) == 2
    assert result["matches"][0]["yolo_verification"]["status"] == "supported_by_yolo"
    assert (
        q.search("a whale")["matches"][0]["yolo_verification"]["status"]
        == "not_supported"
    )
    assert q.describe()["radio_used"] is False
    from slamibot_fieldmind.store import RecordStore
    import sqlite3

    with RecordStore.open_readonly(
        runtime.config.root, runtime.config.collection_id
    ) as reader:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.connection.execute("DELETE FROM frames")
    assert Image.open(io.BytesIO(q.image(first["frame_id"], True))).size == (64, 48)
    runtime.put_place(
        dict(
            place_id="east",
            name="East gate",
            coordinate_id="c1",
            frame="camera_init",
            x=0,
            y=0,
            z=0,
            radius_m=0.1,
        )
    )
    assert len(q.find_frames(place_id="east")["frames"]) == 1
    assert (
        q.describe("location", place_id="east")["views"][0]["frame_id"]
        == first["frame_id"]
    )
    runtime.session(
        dict(session_id="s2", coordinate_id="c2", task_id="task2", state="running")
    )
    third = runtime.ingest(
        observation(
            1,
            1_000_000_000,
            session_id="s2",
            coordinate_id="c2",
            captured_at=first["captured_at"],
        ),
        jpeg(),
    )
    assert third["frame_id"] != first["frame_id"]
    wait_for(lambda: q.get_frame(third["frame_id"])["radio_status"] == "ready")
    assert (
        q.compare_frames([first["frame_id"], third["frame_id"]])["frames"][1][
            "distance_from_first_m"
        ]
        is None
    )
    assert len(q.find_frames(place_id="east")["frames"]) == 1


def test_missing_pose_and_detector_are_not_empty_scene(runtime):
    q = BackendQuery(runtime)
    m = observation(
        camera_pose=None, body_pose=None, detection_status="unavailable", detections=[]
    )
    result = runtime.ingest(m, jpeg())
    assert result["camera_pose"] is None
    assert q.describe()["status"] == "detection_unavailable"
    assert q.describe()["views"][0]["label_counts"] is None
    wait_for(lambda: q.get_frame(result["frame_id"])["radio_status"] == "ready")
    assert (
        q.search("person")["matches"][0]["yolo_verification"]["status"] == "unavailable"
    )
    with pytest.raises(ValueError):
        q.find_frames(near_frame_id=result["frame_id"], radius_m=3)
    runtime.ingest(observation(2, 4_000_000_000, detections=[]), jpeg())
    assert q.describe()["views"][0]["label_counts"] == {}


def test_current_evidence_uses_unsaved_live_frame_and_same_metadata(runtime):
    q = BackendQuery(runtime)
    first = runtime.ingest(observation(), jpeg())
    output = io.BytesIO()
    Image.new("RGB", (64, 48), "blue").save(output, "JPEG")
    skipped = runtime.ingest(
        observation(2, 10_000_000_000, detections=[]), output.getvalue()
    )
    assert skipped["status"] == "skipped"
    result, images = q.describe_evidence()
    assert result["views"][0]["source_sequence"] == 2
    assert result["views"][0]["label_counts"] == {}
    assert images[0]["context"]["frame_id"] == result["views"][0]["frame_id"]
    assert images[0]["context"]["frame_id"] != first["frame_id"]
    assert images[0]["context"]["source"] == "live_cache"
    r, g, b = Image.open(io.BytesIO(images[0]["jpeg"])).getpixel((20, 20))
    assert b > 200 and r < 20 and g < 20
    assert len(list(runtime.config.root.glob("images/**/*.jpg"))) == 1


def test_detector_failure_does_not_prevent_visual_scene_analysis(runtime):
    runtime.ingest(observation(detection_status="unavailable", detections=[]), jpeg())
    result, images = BackendQuery(runtime).describe_evidence()
    assert result["status"] == "detection_unavailable"
    assert len(images) == 1
    assert result["visual_review"]["status"] == "required"
    assert result["visual_review"]["performed"] is False


def test_search_attaches_bounded_unverified_images_and_handles_missing_evidence(
    runtime,
):
    q = BackendQuery(runtime)
    ids = [
        runtime.ingest(observation(i, i * 1_000_000_000, x=i), jpeg())["frame_id"]
        for i in range(1, 6)
    ]
    wait_for(lambda: runtime.health()["indexed_frames"] == 5)
    result, images = q.search_evidence(
        query="a fire truck", top_k=5, group_results=False
    )
    assert result["candidate_only"] is True
    assert len(result["matches"]) == 5 and len(images) == 3
    assert all(
        row["visual_verification"] == "not_performed" for row in result["matches"]
    )
    assert [row["image_evidence"]["status"] for row in result["matches"]] == [
        "attached"
    ] * 3 + ["omitted_limit"] * 2
    # A missing historical JPEG must not turn ranking into visual confirmation.
    image_path = q.get_frame(ids[0])["image_path"]
    (runtime.config.root / image_path).unlink()
    result, images = q.search_evidence(
        query="a fire truck", top_k=5, group_results=False
    )
    row = next(row for row in result["matches"] if row["frame_id"] == ids[0])
    assert row["image_evidence"]["status"] == "unavailable"
    assert row["visual_verification"] == "not_performed"
    result, images = q.search_evidence(query="a fire truck", include_images=False)
    assert not images and result["visual_review"]["status"] == "not_requested"


def test_stale_image_cannot_be_misrepresented_as_current(runtime):
    old = "2001-01-01T00:00:00+00:00"
    runtime.ingest(observation(captured_at=old), jpeg())
    result, images = BackendQuery(runtime).describe_evidence()
    assert result["status"] == "stale"
    assert images[0]["context"]["stale"] is True
    assert images[0]["context"]["captured_at"].startswith("2001-")


def test_task_stop_preserves_live_cache(runtime):
    runtime.session(
        dict(session_id="s1", coordinate_id="c1", task_id="task1", state="stopped")
    )
    assert runtime.ingest(observation(), jpeg())["status"] == "skipped"
    assert BackendQuery(runtime).describe()["status"] == "ok"
    assert runtime.health()["indexed_frames"] == 0


def test_timezone_filters_and_directory_exclusion(runtime):
    import datetime as dt

    result = runtime.ingest(observation(), jpeg())
    moment = dt.datetime.fromisoformat(result["captured_at"]).astimezone(
        dt.timezone(dt.timedelta(hours=8))
    )
    rows = BackendQuery(runtime).find_frames(
        start_time=(moment - dt.timedelta(seconds=1)).isoformat(),
        end_time=(moment + dt.timedelta(seconds=1)).isoformat(),
    )
    assert rows["returned"] == 1
    with pytest.raises(RuntimeError, match="another backend"):
        BackendRuntime(runtime.config, Encoder)


def test_model_failure_does_not_hide_current_input(tmp_path):
    class Broken(Encoder):
        def encode_images(self, images):
            raise RuntimeError("intentional inference failure")

    r = BackendRuntime(
        BackendConfig(root=tmp_path, min_free_bytes=0, require_mount=False), Broken
    )
    try:
        wait_for(lambda: r.model_state == "ready")
        r.session(
            dict(session_id="s1", coordinate_id="c1", task_id="task1", state="running")
        )
        result = r.ingest(observation(), jpeg())
        q = BackendQuery(r)
        wait_for(lambda: q.get_frame(result["frame_id"])["radio_status"] == "failed")
        assert (
            "intentional inference failure"
            in q.get_frame(result["frame_id"])["radio_error"]
        )
        assert q.describe()["status"] == "ok"
        assert q.search("person")["matches"] == []
    finally:
        r.close()


def test_worker_failure_is_reported_and_text_queries_fail_promptly(
    tmp_path, monkeypatch
):
    from slamibot_fieldmind.semantic_store import ObservationStore

    def broken_pending(self, limit=4):
        raise OSError("intentional journal read failure")

    monkeypatch.setattr(ObservationStore, "pending", broken_pending)
    r = BackendRuntime(
        BackendConfig(root=tmp_path, min_free_bytes=0, require_mount=False), Encoder
    )
    try:
        wait_for(lambda: r.model_state == "failed")
        assert "intentional journal" in r.health()["model_error"]
        with pytest.raises(RuntimeError, match="intentional journal"):
            r.encode_text("person")
        assert r.ingest(observation(), jpeg())["status"] == "skipped"
        assert BackendQuery(r).describe()["status"] == "ok"
    finally:
        r.close()


def test_stdio_proxy_roundtrip_over_real_http(runtime):
    import asyncio
    import socket
    import uvicorn
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    result = runtime.ingest(observation(), jpeg())
    wait_for(
        lambda: (
            BackendQuery(runtime).get_frame(result["frame_id"])["radio_status"]
            == "ready"
        )
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(runtime, close_runtime=False), log_level="error")
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        wait_for(lambda: server.started)

        async def call():
            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "slamibot_fieldmind.mcp_proxy",
                    "--url",
                    f"http://127.0.0.1:{port}/mcp",
                ],
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    assert len(tools) == 9
                    response = await session.call_tool(
                        "search_frames", {"query": "a person"}
                    )
                    assert not response.is_error
                    assert (
                        json.loads(response.content[0].text)["matches"][0]["frame_id"]
                        == result["frame_id"]
                    )
                    assert any(item.type == "image" for item in response.content)
                    assert (
                        response.structured_content["visual_review"]["performed"]
                        is False
                    )
                    response = await session.call_tool("describe_view", {})
                    assert any(item.type == "image" for item in response.content)
                    assert (
                        response.structured_content["views"][0]["frame_id"]
                        == result["frame_id"]
                    )
                    response = await session.call_tool(
                        "get_frame_image", {"frame_id": result["frame_id"]}
                    )
                    assert not response.is_error
                    assert response.content[0].type == "image"

        asyncio.run(call())
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def test_vector_append_after_interrupted_sidecar_update(tmp_path):
    from slamibot_fieldmind.store import RecordStore
    from slamibot_fieldmind.pose import Pose
    import datetime as dt

    with RecordStore(tmp_path, "test", "space", 2) as s:
        s.register_camera("CAM_A")
        first = s.add(
            "CAM_A",
            dt.datetime.now(dt.timezone.utc),
            1,
            np.array([1, 0]),
            Pose(0, 0, 0, 0, 0, 0, 1),
        )
        path = s.partition_path("CAM_A", first.vector_partition)
        with path.open("ab") as handle:
            handle.write(np.array([0, 1], dtype=np.float16).tobytes())
        second = s.add(
            "CAM_A",
            dt.datetime.now(dt.timezone.utc),
            2,
            np.array([1, 0]),
            Pose(0, 0, 0, 0, 0, 0, 1),
        )
        assert second.vector_row == 2
        np.testing.assert_allclose(s.load_vectors([second]), [[1, 0]])
        assert not s.verify()["ok"], (
            "unreferenced tail must remain visible, not silently repaired"
        )


def test_http_protocol_and_mcp_schema(runtime):
    with TestClient(
        create_app(runtime, close_runtime=False), base_url="http://127.0.0.1:8765"
    ) as client:
        r = client.post(
            "/v1/observations",
            data={"metadata": json.dumps(observation())},
            files={"image": ("frame.jpg", jpeg(), "image/jpeg")},
        )
        assert r.status_code == 200, r.text
        frame_id = r.json()["frame_id"]
        assert (
            client.get(f"/v1/frames/{frame_id}/image").headers["content-type"]
            == "image/jpeg"
        )
        assert client.get("/v1/health").json()["stores_images"]
        assert (
            client.post(
                "/v1/observations",
                data={"metadata": "{}"},
                files={"image": ("frame.jpg", b"bad", "image/jpeg")},
            ).status_code
            == 400
        )
        # Real Streamable HTTP initialization and tool enumeration over ASGI.
        init = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert init.status_code == 200, init.text
        sid = init.headers["mcp-session-id"]
        headers = {
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": sid,
        }
        tools = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=headers,
        )
        assert tools.status_code == 200, tools.text
        text = tools.text
        for name in (
            "search_frames",
            "describe_view",
            "get_frame_image",
            "list_sessions",
            "list_places",
        ):
            assert name in text
        for name, args in (
            ("describe_view", {}),
            ("find_frames", {}),
            ("search_frames", {"query": "a person"}),
            ("get_frame_image", {"frame_id": frame_id}),
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args},
                },
                headers=headers,
            )
            assert response.status_code == 200, response.text
            assert '"isError":true' not in response.text, response.text


def test_backlog_low_disk_and_restart_recovery(tmp_path):
    gate = threading.Event()

    class SlowEncoder(Encoder):
        def encode_images(self, images):
            gate.wait(5)
            return super().encode_images(images)

    cfg = BackendConfig(
        root=tmp_path, min_free_bytes=0, require_mount=False, pending_limit=1
    )
    r = BackendRuntime(cfg, SlowEncoder)
    wait_for(lambda: r.model_state == "ready")
    r.session(
        dict(session_id="s1", coordinate_id="c1", task_id="task1", state="running")
    )
    first = r.ingest(observation(), jpeg())
    assert (
        r.ingest(observation(2, 4_000_000_000, x=1), jpeg())["status"] == "overloaded"
    )
    assert BackendQuery(r).describe()["status"] == "ok"
    gate.set()
    wait_for(
        lambda: BackendQuery(r).get_frame(first["frame_id"])["radio_status"] == "ready"
    )
    r.config.min_free_bytes = 10**30
    assert r.ingest(observation(3, 7_000_000_000, x=2), jpeg())["status"] == "low_disk"
    r.close()

    # A pending image survives process/model failure and is recovered on restart.
    class Broken:
        def __init__(self):
            raise RuntimeError("model unavailable")

    cfg.min_free_bytes = 0
    r = BackendRuntime(cfg, Broken)
    wait_for(lambda: r.model_state == "failed")
    pending = r.ingest(observation(4, 10_000_000_000, x=3), jpeg())
    assert pending["status"] == "accepted"
    r.close()
    r = BackendRuntime(cfg, Encoder)
    try:
        wait_for(
            lambda: (
                BackendQuery(r).get_frame(pending["frame_id"])["radio_status"]
                == "ready"
            )
        )
    finally:
        r.close()


def test_ros_math_pairing_and_decoder_match_verified_parser():
    base = Path(__file__).resolve().parents[3]
    candidates = [
        base / path / "src/semantic_perception/src"
        for path in ("D360/slamibot_d360_framework", "0801/SLAMIBOT_D360_Framework")
    ]
    bridge = next((path for path in candidates if path.exists()), candidates[0])
    if not bridge.exists():
        pytest.skip("framework adapter not checked out")
    sys.path.insert(0, str(bridge))
    from semantic_bridge.core import PoseBuffer, camera_pose, quaternion_matrix
    from semantic_bridge.oak_yolo import FramePairer, decode_heads

    buffer = PoseBuffer()
    buffer.add(0, pose())
    buffer.add(100_000_000, pose(1, 90))
    p, delta = buffer.at(50_000_000)
    assert p["x"] == 0.5 and delta == 0.05
    np.testing.assert_allclose(
        quaternion_matrix([p[k] for k in ("qx", "qy", "qz", "qw")]),
        quaternion_matrix([0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)]),
        atol=1e-6,
    )
    assert buffer.at(300_000_000) == (None, None)
    assert buffer.add(1, pose()) is True
    transform = np.eye(4)
    transform[0, 3] = 1
    cp = camera_pose(pose(angle=90), transform)
    assert abs(cp["x"]) < 1e-6 and abs(cp["y"] - 1) < 1e-6
    simulated = dict(pose(), source="simulated")
    assert camera_pose(simulated, transform)["source"] == "simulated"
    assert camera_pose(pose(), None) is None
    clock = [0.0]
    pairer = FramePairer(clock=lambda: clock[0])
    pairer.put("images", 1, "one")
    pairer.put("selected", 1, True)
    pairer.put("results", 2, [])
    assert pairer.poll() == []
    clock[0] = 0.3
    assert pairer.poll() == [(1, "one", None)]
    heads = [np.zeros((1, 85, 288 // s, 512 // s), np.float32) for s in (8, 16, 32)]
    heads[0][0, 0:5, 10, 10] = [0.5, 0.5, 1, 1, 0.9]
    heads[0][0, 5, 10, 10] = 0.95
    labels = ["person"] + [f"c{i}" for i in range(79)]
    output = decode_heads(heads, labels)
    from depthai_nodes.node.parsers.utils.yolo import (
        compute_yolo_detections,
        YOLOSubtype,
    )

    reference = compute_yolo_detections(
        subtype=YOLOSubtype.V6R1,
        layer_names=["259", "483", "707"],
        outputs_values=[h.copy() for h in heads],
        strides=[8, 16, 32],
        conf_threshold=0.2,
        n_classes=80,
        iou_threshold=0.45,
        max_det=300,
        anchors=None,
        n_keypoints=0,
        label_names=labels,
        keypoint_label_names=None,
        keypoint_edges=None,
        input_shape=(288, 512),
    )
    cx, cy, w, h = reference["bboxes"][0]
    expected = 0.15 + 0.7 * np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    np.testing.assert_allclose(output[0]["bbox_norm"], expected, atol=1e-6)
