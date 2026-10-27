"""Device-side end-to-end smoke test for the FieldMind backend.

Exercises the real services this deployment runs: HTTP session creation, JPEG
observation ingest, keyframe persistence, RADIO text indexing on the actual
hardware, and every MCP tool through the backend's streamable HTTP endpoint.

Run it while the backend is already listening, or let it start one itself:

    python scripts/device_smoke.py --root /mnt/usb/slamibot_record_data/semantic_index_smoke \
        --model-config configs/jetson.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def jpeg_bytes(path: Path | None, size=(1024, 576)) -> bytes:
    if path is not None:
        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()
    # Deterministic synthetic scene so the run does not need a camera or a bag.
    import numpy as np

    width, height = size
    y, x = np.mgrid[0:height, 0:width]
    frame = np.stack(
        [
            (127 + 100 * np.sin(x / 40.0)).astype(np.int16),
            (127 + 100 * np.cos(y / 30.0)).astype(np.int16),
            ((x * y / (width * height)) * 255).astype(np.int16),
        ],
        axis=-1,
    )
    buffer = io.BytesIO()
    Image.fromarray(np.clip(frame, 0, 255).astype("uint8")).save(
        buffer, format="JPEG", quality=85
    )
    return buffer.getvalue()


def post_observation(base: str, session: dict, sequence: int, jpeg: bytes) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    metadata = dict(
        protocol_version=1,
        camera_id="CAM_A",
        device_id="orin-nx-smoke",
        source_id=session["session_id"],
        source_sequence=sequence,
        source_stamp_ns=int(now.timestamp() * 1e9),
        captured_at=now.isoformat(timespec="milliseconds"),
        received_at=now.isoformat(timespec="milliseconds"),
        timestamp_source="host_receive",
        sync_quality="unknown",
        session_id=session["session_id"],
        coordinate_id=session["coordinate_id"],
        body_pose={
            "x": sequence * 0.7,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "frame": "map",
            "source": "simulated",
        },
        camera_pose={
            "x": sequence * 0.7,
            "y": 0.0,
            "z": 0.5,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "frame": "map",
            "source": "simulated",
        },
        detection_status="available",
        detections=[
            dict(label="person", label_id=0, confidence=0.9, bbox_norm=[0.1, 0.2, 0.4, 0.8])
        ],
    )
    boundary = "----fieldmind-smoke"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="metadata"\r\n\r\n')
    body.write(json.dumps(metadata).encode())
    body.write(f"\r\n--{boundary}\r\n".encode())
    body.write(
        b'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
    )
    body.write(jpeg)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    payload = body.getvalue()
    request = urllib.request.Request(
        f"{base}/v1/observations",
        data=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload)),
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def post_json(url: str, value: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


async def mcp_checks(url: str, frame_id: str, report: dict) -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = [tool.name for tool in (await session.list_tools()).tools]
            report["mcp_tools"] = tools

            async def call(name, **arguments):
                result = await session.call_tool(name, arguments)
                if result.is_error:
                    raise RuntimeError(f"{name}: {result.content}")
                return result, json.loads(result.content[0].text)

            _, status = await call("index_status")
            report["index_status"] = status
            _, sessions = await call("list_sessions")
            report["list_sessions"] = sessions
            _, places = await call("list_places")
            report["list_places"] = places
            _, found = await call("find_frames", limit=10)
            report["find_frames"] = found
            _, search = await call("search_frames", query="a person standing outside", top_k=3)
            report["search_frames"] = search
            _, view = await call("describe_view")
            report["describe_view"] = view
            frames = found.get("frames") or report["search_frames"].get("frames") or []
            if not frames:
                raise RuntimeError("no frames were indexed, cannot check frame tools")
            target = frames[0]["frame_id"]
            _, frame = await call("get_frame", frame_id=target)
            report["get_frame"] = frame
            image_result = await session.call_tool(
                "get_frame_image", dict(frame_id=target, overlay=True)
            )
            if image_result.is_error:
                raise RuntimeError(f"get_frame_image: {image_result.content}")
            images = [c for c in image_result.content if c.type == "image"]
            report["image_bytes"] = len(base64.b64decode(images[0].data)) if images else 0
            if len(frames) >= 2:
                _, compare = await call(
                    "compare_frames", frame_ids=[f["frame_id"] for f in frames[:2]]
                )
                report["compare_frames"] = compare


def wait_health(base: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            return get_json(f"{base}/v1/health")
        except Exception as error:  # noqa: BLE001 - still starting
            last = error
            time.sleep(1.0)
    raise RuntimeError(f"backend did not become healthy: {last}")


def wait_ready(base: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_json(f"{base}/v1/health")
        if status.get("model_state") in ("ready", "failed"):
            return status
        time.sleep(2.0)
    raise RuntimeError("model did not finish loading")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "jetson.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "device_smoke.json")
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    started = None
    try:
        try:
            get_json(f"{base}/v1/health")
        except Exception:  # noqa: BLE001 - not running yet
            args.root.mkdir(parents=True, exist_ok=True)
            log = open(ROOT / "reports" / "device_smoke_backend.log", "w")
            started = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "slamibot_fieldmind.backend",
                    "--root",
                    str(args.root),
                    "--model-config",
                    str(args.model_config),
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        report: dict = dict(started_at=dt.datetime.now(dt.timezone.utc).isoformat())
        report["health_initial"] = wait_health(base, 60)
        report["health_ready"] = wait_ready(base, 900)
        if report["health_ready"].get("model_state") != "ready":
            report["error"] = report["health_ready"].get("model_error")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(json.dumps(report, indent=2)[:2000])
            return 1

        session = dict(
            session_id="smoke-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S"),
            coordinate_id="smoke-map",
            task_id="smoke-task",
            state="running",
            mode="device_smoke",
            pose_source="simulated",
        )
        report["session"] = post_json(f"{base}/v1/sessions", session)

        jpeg = jpeg_bytes(args.image)
        accepted = []
        t_start = time.monotonic()
        for index in range(args.frames):
            result = post_observation(base, session, index, jpeg)
            accepted.append(result.get("status"))
            if index + 1 < args.frames:
                time.sleep(0.6)
            jpeg = jpeg_bytes(args.image, size=(1024, 576)) if args.image else jpeg
        report["ingest_status"] = accepted
        report["ingest_seconds"] = round(time.monotonic() - t_start, 2)

        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            status = get_json(f"{base}/v1/health")
            if status.get("states", {}).get("pending", 0) == 0:
                break
            time.sleep(2.0)
        report["health_after"] = get_json(f"{base}/v1/health")

        found = get_json(f"{base}/v1/health")  # touch again after indexing settles
        report["health_confirm"] = found
        frames = report["health_after"].get("states", {})
        report["indexed_ready"] = frames.get("ready", 0)

        asyncio.run(mcp_checks(f"{base}/mcp", "unused", report))
        report["ok"] = True
    finally:
        if started is not None:
            started.terminate()
            try:
                started.wait(timeout=30)
            except subprocess.TimeoutExpired:
                started.kill()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = dict(
        ok=report.get("ok"),
        model_state=report["health_ready"].get("model_state"),
        ingest=report.get("ingest_status"),
        states=report.get("health_after", {}).get("states"),
        mcp_tools=len(report.get("mcp_tools", [])),
        search_matches=len(report.get("search_frames", {}).get("matches", [])),
        image_bytes=report.get("image_bytes"),
        report=str(args.out),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
