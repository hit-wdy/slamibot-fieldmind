"""Persistent ROS-independent backend: python -m slamibot_fieldmind.backend --help."""

from __future__ import annotations

import argparse
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .backend_query import BackendQuery
from .backend_runtime import BackendConfig, BackendRuntime
from .keyframes import KeyframePolicy
from .protocol import MAX_JPEG_BYTES


def build_backend_mcp(service):
    server = MCPServer(
        name="slamibot-fieldmind",
        instructions=(
            "Visual memory of a mobile robot. Current view comes from the continuous YOLO cache; never open the camera. "
            "Use list_sessions and list_places for coordinate scope. Positions locate the camera that saw an object, not its 3D position. "
            "RADIO cosine is not a probability; unsupported objects return best ranked candidates. Check stale and sync_quality. "
            "Use detections and label_counts for filtered class evidence. detections_original and detections_raw are audit evidence; "
            "never merge them into counts. New YOLOv6 R1 scores use objectness times class score and are not directly comparable to legacy class-only scores."
        ),
    )

    @server.tool(
        description="Rank keyframes using RADIO, then cross-check the intended COCO object class using same-frame YOLO. Cosine is not a probability. Missing YOLO evidence is not proof of absence. English visual phrases are recommended. Spatial filters require an explicit coordinate session."
    )
    def search_frames(
        query: str,
        camera_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        labels: list[str] | None = None,
        top_k: int = 5,
        min_similarity: float | None = None,
        group_results: bool = True,
        group_window_s: float = 2,
        near_frame_id: str | None = None,
        radius_m: float | None = None,
        target_label: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        coordinate_id: str | None = None,
        place_id: str | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        frame: str | None = None,
    ) -> dict:
        values = {k: v for k, v in locals().items() if k != "service"}
        return service.search(**values)

    @server.tool(
        description="Metadata-only lookup of keyframes, including pending/failed RADIO. YOLO labels are evidence, not proof of presence or absence."
    )
    def find_frames(
        labels: list[str] | None = None,
        camera_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        near_frame_id: str | None = None,
        radius_m: float | None = None,
        limit: int = 50,
        task_id: str | None = None,
        session_id: str | None = None,
        coordinate_id: str | None = None,
        place_id: str | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        frame: str | None = None,
    ) -> dict:
        return service.find_frames(
            **{k: v for k, v in locals().items() if k != "service"}
        )

    @server.tool(
        description="Return current YOLO cache (mode=current), or latest saved observations at a registered place (mode=location). No RADIO inference. Quote captured_at and stale; historical observations do not describe the present. Counts are per view."
    )
    def describe_view(
        mode: str = "current",
        location_label: str | None = None,
        camera_id: str | None = None,
        device_id: str | None = None,
        max_age_s: float = 5,
        place_id: str | None = None,
    ) -> dict:
        return service.describe(**{k: v for k, v in locals().items() if k != "service"})

    @server.tool(
        description="Full keyframe detail, camera/body pose, synchronization quality, detection status, RADIO job state and image reference."
    )
    def get_frame(frame_id: str) -> dict:
        return service.get_frame(frame_id)

    @server.tool(
        description="Compare 2..50 keyframes. Distances are unavailable across coordinate sessions or when pose is missing. A simulated legacy pose is a stand-in, not a measured camera position."
    )
    def compare_frames(frame_ids: list[str]) -> dict:
        return service.compare_frames(frame_ids)

    @server.tool(
        description="Coverage, model health, backlog, dropped/overdue sampling, disk and current source freshness. A running process with stale input has not seen the present."
    )
    def index_status() -> dict:
        return service.status()

    @server.tool(
        description="Return stored JPEG evidence for a keyframe. Optional YOLO overlay is class evidence only. Older records may have no image."
    )
    def get_frame_image(
        frame_id: str, overlay: bool = False, max_edge: int = 1280
    ) -> ImageContent:
        return ImageContent(
            type="image",
            mime_type="image/jpeg",
            data=base64.b64encode(service.image(frame_id, overlay, max_edge)).decode(),
        )

    @server.tool(
        description="List SLAM tasks, coordinate sessions and observation coverage before selecting historical filters."
    )
    def list_sessions() -> dict:
        return service.list_sessions()

    @server.tool(
        description="List registered place IDs, names, coordinate scope, centers and radii. Same name in different maps is ambiguous."
    )
    def list_places() -> dict:
        return service.list_places()

    # Preserve the existing public argument name `query` while retaining the closure.
    return server


def create_app(runtime, host="127.0.0.1", close_runtime=True):
    query = BackendQuery(runtime)
    mcp = build_backend_mcp(query)
    app = mcp.streamable_http_app(host=host)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application):
            try:
                yield
            finally:
                if close_runtime:
                    await run_in_threadpool(runtime.close)

    app.router.lifespan_context = lifespan

    async def handle(request: Request):
        try:
            path = request.url.path
            if path == "/v1/health":
                return JSONResponse(await run_in_threadpool(query.status))
            if path == "/v1/sessions":
                result = (
                    await run_in_threadpool(query.list_sessions)
                    if request.method == "GET"
                    else await run_in_threadpool(runtime.session, await request.json())
                )
                return JSONResponse(result)
            if path == "/v1/places":
                result = (
                    await run_in_threadpool(query.list_places)
                    if request.method == "GET"
                    else await run_in_threadpool(
                        runtime.put_place, await request.json()
                    )
                )
                return JSONResponse(result)
            if path == "/v1/observations":
                length = int(request.headers.get("content-length", "0"))
                if length <= 0 or length > MAX_JPEG_BYTES + 128 * 1024:
                    return JSONResponse(
                        {
                            "error": "Content-Length required; request exceeds size limit"
                        },
                        status_code=413,
                    )
                async with request.form(
                    max_files=1, max_fields=1, max_part_size=MAX_JPEG_BYTES
                ) as form:
                    metadata = json.loads(str(form["metadata"]))
                    jpeg = await form["image"].read(MAX_JPEG_BYTES + 1)
                return JSONResponse(
                    await run_in_threadpool(runtime.ingest, metadata, jpeg)
                )
            if path.endswith("/image"):
                data = await run_in_threadpool(
                    query.image,
                    request.path_params["frame_id"],
                    request.query_params.get("overlay", "false").lower() == "true",
                    int(request.query_params.get("max_edge", 1280)),
                )
                return Response(data, media_type="image/jpeg")
            return JSONResponse(
                await run_in_threadpool(
                    query.get_frame, request.path_params["frame_id"]
                )
            )
        except FileNotFoundError as error:
            return JSONResponse({"error": str(error)}, status_code=404)
        except (ValueError, KeyError, TypeError, OSError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)

    app.routes.extend(
        [
            Route("/v1/health", handle),
            Route("/v1/sessions", handle, methods=["GET", "POST"]),
            Route("/v1/places", handle, methods=["GET", "PUT"]),
            Route("/v1/observations", handle, methods=["POST"]),
            Route("/v1/frames/{frame_id}/image", handle),
            Route("/v1/frames/{frame_id}", handle),
        ]
    )
    app.state.runtime = runtime
    app.state.query = query
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=BackendConfig.__dataclass_fields__["root"].default
    )
    p.add_argument("--collection-id", default="radio_clip_v1")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model-config", type=Path)
    p.add_argument(
        "--allow-unmounted",
        action="store_true",
        help="explicit local development override",
    )
    p.add_argument("--min-free-gib", type=float, default=2)
    p.add_argument("--distance-m", type=float, default=0.5)
    p.add_argument("--rotation-deg", type=float, default=15)
    p.add_argument("--max-interval-s", type=float, default=3)
    p.add_argument("--min-interval-s", type=float, default=0.5)
    args = p.parse_args(argv)
    from .config import Config
    from .encoder import AlignedEncoder

    cfg = Config.from_yaml(args.model_config) if args.model_config else Config.default()
    runtime = BackendRuntime(
        BackendConfig(
            root=args.root,
            collection_id=args.collection_id,
            min_free_bytes=int(args.min_free_gib * 1024**3),
            require_mount=not args.allow_unmounted,
            policy=KeyframePolicy(
                args.distance_m,
                args.rotation_deg,
                args.max_interval_s,
                args.min_interval_s,
            ),
        ),
        lambda: AlignedEncoder(cfg),
    )
    import uvicorn

    try:
        uvicorn.run(
            create_app(runtime, args.host), host=args.host, port=args.port, workers=1
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
