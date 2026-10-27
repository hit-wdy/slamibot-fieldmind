"""Persistent ROS-independent backend: python -m slamibot_fieldmind.backend --help."""

from __future__ import annotations

import argparse
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .backend_query import BackendQuery
from .backend_runtime import BackendConfig, BackendRuntime
from .keyframes import KeyframePolicy
from .protocol import MAX_JPEG_BYTES


def multimodal_result(result, images):
    content = [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    for image in images:
        content.append(
            TextContent(
                type="text",
                text="VISUAL_EVIDENCE "
                + json.dumps(image["context"], ensure_ascii=False),
            )
        )
        content.append(
            ImageContent(
                type="image",
                mime_type="image/jpeg",
                data=base64.b64encode(image["jpeg"]).decode(),
            )
        )
    return CallToolResult(content=content, structured_content=result)


def build_backend_mcp(service):
    server = MCPServer(
        name="slamibot-fieldmind",
        instructions=(
            "Visual memory of a mobile robot. describe_view returns atomic current image/YOLO snapshots; never open the camera. "
            "search_frames and describe_view include raw evidence images by default. The calling multimodal agent MUST inspect them before confirming objects or describing a scene. "
            "Reject visually mismatched retrievals even when they rank first. If none match, report no confirmed evidence among reviewed candidates. "
            "Use list_sessions and list_places for coordinate scope. Positions locate the camera that saw an object, not its 3D position. "
            "RADIO cosine is not a probability; unsupported objects return best ranked candidates. Check stale and sync_quality. "
            "Use detections and label_counts for filtered class evidence. detections_original and detections_raw are audit evidence; "
            "never merge them into counts. New YOLOv6 R1 scores use objectness times class score and are not directly comparable to legacy class-only scores."
        ),
    )

    @server.tool(
        description="Retrieve historical candidates with RADIO and YOLO class evidence, plus up to 3 raw images by default. These are UNVERIFIED candidates: inspect images to confirm the requested object, attributes and counts; reject mismatches. Highest cosine does not imply existence. If none match, say no confirmed evidence in reviewed candidates, not absence everywhere. English query phrases are recommended."
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
        include_images: bool = True,
        max_edge: int = 1280,
    ) -> CallToolResult:
        values = {k: v for k, v in locals().items() if k != "service"}
        return multimodal_result(*service.search_evidence(**values))

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
        description="Return current raw images and same-observation YOLO metadata (mode=current), or latest historical views at a place (mode=location). Up to 3 images included by default, even if no historical keyframe or YOLO failed. Describe the full image with your vision capability, then distinguish YOLO detections and disagreements. Check captured_at/stale. No RADIO or VLM runs inside this tool."
    )
    def describe_view(
        mode: str = "current",
        location_label: str | None = None,
        camera_id: str | None = None,
        device_id: str | None = None,
        max_age_s: float = 5,
        place_id: str | None = None,
        include_images: bool = True,
        max_edge: int = 1280,
    ) -> CallToolResult:
        return multimodal_result(
            *service.describe_evidence(
                **{k: v for k, v in locals().items() if k != "service"}
            )
        )

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


def _encoder_factory(encoder_url, encoder_timeout, cfg, aligned_encoder_cls):
    """Pick the in-process encoder or the GPU sidecar client.

    A sidecar is the only way to use the GPU on JetPack 5, where the CUDA torch
    wheels are CPython 3.8 and this backend needs CPython 3.10+.
    """
    if not encoder_url:
        return lambda: aligned_encoder_cls(cfg)

    from .remote_encoder import connect

    def load():
        # The sidecar has its own model config; the fingerprint check in
        # `connect` plus `bind_encoder` refuse a collection from another space.
        return connect(encoder_url, timeout=encoder_timeout)

    return load


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
    p.add_argument(
        "--mount-path",
        type=Path,
        help="mounted volume that must contain --root (default: /mnt/usb)",
    )
    p.add_argument(
        "--encoder-url",
        help=(
            "serve the model from an encoder sidecar at this URL instead of loading it "
            "in-process (see python -m slamibot_fieldmind.encoder_service)"
        ),
    )
    p.add_argument(
        "--encoder-timeout",
        type=float,
        default=120,
        help="seconds to wait for the sidecar to load and for each request",
    )
    p.add_argument("--min-free-gib", type=float, default=2)
    p.add_argument("--distance-m", type=float, default=0.5)
    p.add_argument("--rotation-deg", type=float, default=15)
    p.add_argument("--max-interval-s", type=float, default=3)
    p.add_argument("--min-interval-s", type=float, default=0.5)
    p.add_argument(
        "--timeout-distance-m",
        type=float,
        default=0.05,
        help="minimum translation for time-triggered keyframes; suppresses static jitter",
    )
    p.add_argument(
        "--timeout-rotation-deg",
        type=float,
        default=2,
        help="minimum 3D rotation for time-triggered keyframes",
    )
    args = p.parse_args(argv)
    from .config import Config
    from .encoder import AlignedEncoder

    cfg = Config.from_yaml(args.model_config) if args.model_config else Config.default()
    encoder_factory = _encoder_factory(args.encoder_url, args.encoder_timeout, cfg, AlignedEncoder)
    runtime = BackendRuntime(
        BackendConfig(
            root=args.root,
            collection_id=args.collection_id,
            min_free_bytes=int(args.min_free_gib * 1024**3),
            require_mount=not args.allow_unmounted,
            mount_path=args.mount_path,
            policy=KeyframePolicy(
                args.distance_m,
                args.rotation_deg,
                args.max_interval_s,
                args.min_interval_s,
                args.timeout_distance_m,
                args.timeout_rotation_deg,
            ),
        ),
        encoder_factory,
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
