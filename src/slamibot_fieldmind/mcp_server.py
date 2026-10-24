"""MCP server over the record collection.

A thin wrapper over QueryService for retrieval and SceneService for YOLO scene
inventory. An HTTP surface can wrap the same services later.

Spec section 9 requires the tool descriptions themselves to carry the caveats,
because a harness reads those and not this docstring. They are in each `tool`
call below.

There is deliberately no `get_frame_image`. The requirement is that frames are
never stored, so no tool can return one, and offering a tool that always fails
would be worse than not offering it.

Transports: `stdio` (default, for a local harness) or `streamable-http`
(bind to 127.0.0.1 unless you have a reason not to).

    python -m slamibot_fieldmind.mcp_server --root data/records
    python -m slamibot_fieldmind.mcp_server --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .encoder import AlignedEncoder
from .paths import PROJECT_ROOT
from .query import QueryService
from .scene import SceneService

DEFAULT_ROOT = PROJECT_ROOT / "data" / "records"
DEFAULT_COLLECTION = "radio_clip_v1"

CAVEATS = (
    "similarity is a cosine in the RADIO/CLIP space, not a probability, and no threshold has "
    "been calibrated on labelled data. These are ranked candidates: never present them as "
    "detection or identification. "
    "If the index has no recent records, say the index is stale rather than that nothing was there. "
    "A time group is a set of candidates close in time, not an event boundary."
)


def build_server(service: QueryService, scene_service: SceneService | None = None) -> MCPServer:
    scenes = scene_service or SceneService(service.root, service.collection_id)
    server = MCPServer(
        name="slamibot-fieldmind",
        title="RADIO camera retrieval",
        instructions=(
            "Search and compare camera records. Each record holds a RADIO/CLIP embedding, YOLO "
            "detections and a camera pose for one sampled instant. No images are stored, so no "
            "tool returns one. Use describe_view for 'what is in view / at a place'; it returns YOLO "
            "detections without RADIO. Current mode opens the local camera; location mode reads the "
            "latest stored observation and reports freshness. " + CAVEATS
        ),
    )

    @server.tool(
        name="search_frames",
        description=(
            "Rank stored frames against an English visual phrase (for example 'a fire truck'). "
            "Returns candidate frames with their frame id, camera, location label, similarity, "
            "detections, pose and time. Each match carries `pose_is_a_stand_in` when the pose is "
            "simulated: quote the location label and the frame id as where and when, and do not "
            "present simulated coordinates as a real position. "
            "Filtering by camera, time, detection label or a radius around a known frame happens "
            "before ranking. Use RADIO retrieval first: pass target_label for the intended object "
            "(an exact COCO class such as 'person' or 'bed', or an unsupported name such as 'ceiling light'). "
            "Each match includes yolo_verification: supported_by_yolo, not_detected, or not_supported. "
            "YOLO evidence never removes or reorders RADIO candidates. For unsupported objects, return "
            "the highest-ranked RADIO candidate with its similarity. Do not set labels merely to verify "
            "a query: labels is a hard pre-filter. Simple class-name queries infer their own target; "
            "descriptive queries require target_label to verify the intended object class. " + CAVEATS
        ),
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
        group_window_s: float = 2.0,
        near_frame_id: str | None = None,
        radius_m: float | None = None,
        target_label: str | None = None,
    ) -> dict:
        """Search frames by a visual phrase.

        Args:
            query: An English visual phrase. Chinese queries are not reliably supported;
                rewrite them into English first.
            camera_ids: Restrict to these cameras.
            start_time: ISO 8601 lower bound. Defaults to 24 hours before end_time.
            end_time: ISO 8601 upper bound, exclusive. Defaults to now.
            labels: Restrict to frames where YOLO reported any of these COCO class names.
            top_k: How many results to return, 1 to 50.
            min_similarity: Optional cosine floor. Leave unset to get ranked candidates.
            group_results: Merge candidates close in time on one camera.
            group_window_s: Grouping window in seconds.
            near_frame_id: Restrict to a radius around this frame's pose.
            radius_m: Radius in metres, required with near_frame_id.
            target_label: Object to cross-check after RADIO retrieval. Use an exact COCO class
                for supported objects, or the unsupported object name for RADIO-only results.
                For 'a person wearing glasses', pass 'person'; YOLO only checks the person class.
                If omitted, only simple class-name queries such as 'a person' are recognised.
        """
        return service.search(
            query=query,
            camera_ids=camera_ids,
            start_time=start_time,
            end_time=end_time,
            labels=labels,
            top_k=top_k,
            min_similarity=min_similarity,
            group_results=group_results,
            group_window_s=group_window_s,
            near_frame_id=near_frame_id,
            radius_m=radius_m,
            target_label=target_label,
        )

    @server.tool(
        name="find_frames",
        description=(
            "Look up frames by metadata alone: YOLO detection labels, camera, time range, or a "
            "radius around a known frame. No text query and no model, so this is fast. "
            "Detection labels come from a YOLO detector and carry false positives and frame-to-frame "
            "flicker; a frame without a label is not proof the object was absent."
        ),
    )
    def find_frames(
        labels: list[str] | None = None,
        camera_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        near_frame_id: str | None = None,
        radius_m: float | None = None,
        limit: int = 50,
    ) -> dict:
        """Find frames by detection label, time, camera or position.

        Args:
            labels: COCO class names such as 'truck', 'person', 'laptop'.
            camera_ids: Restrict to these cameras.
            start_time: ISO 8601 lower bound. Defaults to 24 hours before end_time.
            end_time: ISO 8601 upper bound, exclusive. Defaults to now.
            near_frame_id: Restrict to a radius around this frame's pose.
            radius_m: Radius in metres, required with near_frame_id.
            limit: Maximum frames to return, up to 1000.
        """
        return service.find_frames(
            labels=labels,
            camera_ids=camera_ids,
            start_time=start_time,
            end_time=end_time,
            near_frame_id=near_frame_id,
            radius_m=radius_m,
            limit=limit,
        )

    @server.tool(
        name="describe_view",
        description=(
            "List what YOLO detected without a text query or RADIO model. mode='current' captures a "
            "fresh local OAK frame and runs YOLO; it needs exclusive camera access and does not store "
            "images or records. If the camera is unavailable, report that; never substitute old records. "
            "mode='location' returns the latest saved detection snapshot per matching camera, using "
            "an exact registered location_label or a camera_id supplied by a navigation caller. "
            "Always quote captured_at and stale; old observations do not describe the present. "
            "Navigation and per-frame place tracking are not connected. YOLO has a limited class "
            "list and centre-crop field of view: empty detections do not mean the scene is empty. "
            "Keep counts per frame/camera; do not add them into a unique-object count."
        ),
    )
    def describe_view(
        mode: str = "current",
        location_label: str | None = None,
        camera_id: str | None = None,
        device_id: str | None = None,
        max_age_s: float = 5.0,
    ) -> dict:
        """Return a live or stored YOLO inventory.

        Args:
            mode: 'current' for live local capture, 'location' for latest saved observations.
            location_label: Exact camera location, such as '小区东门'; only for location mode.
            camera_id: Logical camera ID; navigation may resolve a place to this ID.
            device_id: DepthAI hardware selector, only for current mode.
            max_age_s: Positive freshness limit in seconds; stale records remain explicitly marked.
        """
        return scenes.describe(mode, location_label, camera_id, device_id, max_age_s)

    @server.tool(
        name="get_frame",
        description=(
            "Full detail for one frame id: detections with boxes, pose, and where its embedding "
            "lives. There is no image: records store only the embedding, detections and pose."
        ),
    )
    def get_frame(frame_id: str) -> dict:
        """Return one record in full.

        Args:
            frame_id: A frame id from another tool, e.g. 'camera_01-20261011T153001.738-000366'.
        """
        return service.get_frame(frame_id)

    @server.tool(
        name="compare_frames",
        description=(
            "Compare two or more frames: time gaps, distance and heading change between their "
            "poses, and which detection labels do and do not appear in every frame. "
            "Pose distances are only meaningful when both frames share a coordinate frame, and "
            "simulated poses describe a stand-in trajectory rather than the real camera."
        ),
    )
    def compare_frames(frame_ids: list[str]) -> dict:
        """Compare frames by time, pose and detections.

        Args:
            frame_ids: At least two frame ids from another tool.
        """
        return service.compare_frames(frame_ids)

    @server.tool(
        name="index_status",
        description=(
            "What the index actually covers: frame count per camera, the time range, seconds since "
            "the newest record, which detection labels exist, and whether the vector store matches "
            "the database. Call this before answering 'is there anything right now': a process that "
            "is up with a stale index has not seen the present."
        ),
    )
    def index_status() -> dict:
        """Report index coverage and health."""
        return service.status()

    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="record root")
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    parser.add_argument("--transport", choices=("stdio", "streamable-http", "sse"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="only for http transports")
    parser.add_argument("--port", type=int, default=8765, help="only for http transports")
    parser.add_argument("--preload-encoder", action="store_true", help="load the model at startup instead of on first text query")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    service = QueryService(args.root, args.collection_id, encoder_factory=AlignedEncoder)
    # Fail early with a clear message rather than at the first tool call.
    probe = service.status()
    print(
        f"collection {args.collection_id}: {probe['indexed_frames']} frames, "
        f"{probe['seconds_since_latest_record']}s since the newest record, "
        f"poses are '{probe['pose_source']}', no images stored",
        file=sys.stderr,
    )
    if args.preload_encoder:
        service.encoder()
    server = build_server(service)
    if args.transport == "stdio":
        server.run(transport="stdio")
    elif args.transport == "streamable-http":
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run(transport="sse", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
