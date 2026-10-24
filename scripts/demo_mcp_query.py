"""Run actual MCP queries scoped to the active demo; preserve tool replies separately."""

import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from demo_support import choose_candidates, class_count, relative_place


def pot_pairs(row):
    """Geometry is supporting evidence; it cannot verify that a vase is white."""
    detections = row.get("detections", [])
    pairs = []
    for plant in (d for d in detections if d["label"] == "potted plant"):
        a = plant["bbox_norm"]
        for vase in (d for d in detections if d["label"] == "vase"):
            b = vase["bbox_norm"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area > 0 and intersection / area >= .5:
                pairs.append(dict(plant=plant, vase=vase))
    return pairs


async def query(args):
    manifest = json.loads(args.manifest.read_text("utf-8"))
    session_id = manifest["session"]["session_id"]
    args.out.mkdir(parents=True, exist_ok=False)
    params = StdioServerParameters(command=sys.executable,
        args=["-m", "slamibot_fieldmind.mcp_proxy", "--url", args.url])
    report = dict(intent=args.intent, session_id=session_id, raw_calls=[], candidates=[],
                  selection_stage="agent_side_filter_after_mcp", pose_source="simulated")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()

            async def call(name, **arguments):
                response = await client.call_tool(name, arguments)
                if response.is_error:
                    raise RuntimeError(str(response.content))
                result = json.loads(response.content[0].text)
                report["raw_calls"].append(dict(tool=name, arguments=arguments, response=result))
                return result

            places = await call("list_places")
            points = [p for p in places["places"] if p["coordinate_id"] == manifest["session"]["coordinate_id"]]
            if args.intent == "current":
                result = await call("describe_view", camera_id=manifest["camera_id"])
                views = [v for v in result["views"] if v.get("session_id") == session_id]
                report.update(status="ok" if views else "no_current_view_for_this_session",
                              views=views, radio_used=False, source_status=result["status"],
                              note="Paused replay view, original capture timestamp retained; stale is expected, not a live camera.")
            else:
                phrase, label, count = {
                    "person": ("a person in an indoor hallway or office", "person", None),
                    "four-chairs": ("four chairs in an office", "chair", 4),
                    "white-vase-plant": ("a potted plant in a white vase", "potted plant", None),
                }[args.intent]
                filters = dict(session_id=session_id, start_time=manifest["query_start"],
                               end_time=manifest["query_end"], camera_ids=[manifest["camera_id"]])
                found = await call("search_frames", query=phrase, target_label=label,
                                   top_k=50, group_results=False, **filters)
                selected = choose_candidates(found["matches"], label, count)
                if args.intent == "four-chairs":
                    # Full metadata lookup prevents RADIO top-k from hiding matching counts.
                    all_frames = await call("find_frames", labels=[label], limit=1000, **filters)
                    seen = {r["frame_id"] for r in selected}
                    for row in all_frames["frames"]:
                        if class_count(row, label) == count and row["frame_id"] not in seen:
                            selected.append(dict(row, radio_rank=None, similarity=None,
                                                 candidate_source="find_frames_count_fallback"))
                elif args.intent == "white-vase-plant":
                    # Prefer a detected plant/vase relationship; retain plant-only candidates if absent.
                    paired = [dict(row, plant_vase_pairs=pot_pairs(row)) for row in selected if pot_pairs(row)]
                    if paired:
                        selected = paired
                    report["attribute_verification"] = "White color requires Agent image review; YOLO does not verify color."
                report["candidate_count"] = len(selected)
                # Avoid three copies of nearly the same moment when returning evidence.
                chosen = []
                from datetime import datetime
                for row in selected:
                    stamp = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
                    if any(abs((stamp - datetime.fromisoformat(r["captured_at"].replace("Z", "+00:00"))).total_seconds()) < 2 for r in chosen):
                        continue
                    chosen.append(row)
                    if len(chosen) == args.images:
                        break
                for row in chosen:
                    frame = await call("get_frame", frame_id=row["frame_id"])
                    if frame.get("status") != "ok":
                        raise RuntimeError("Selected frame no longer exists")
                    response = await client.call_tool("get_frame_image", dict(frame_id=row["frame_id"], overlay=True))
                    if response.is_error:
                        raise RuntimeError(str(response.content))
                    block = next(b for b in response.content if b.type == "image")
                    path = args.out / (row["frame_id"] + ".jpg")
                    path.write_bytes(base64.b64decode(block.data))
                    report["raw_calls"].append(dict(tool="get_frame_image", arguments=dict(frame_id=row["frame_id"], overlay=True),
                                                    image_file=str(path.resolve())))
                    report["candidates"].append(dict(frame_id=row["frame_id"], captured_at=row["captured_at"],
                        radio_similarity=row.get("similarity"), original_radio_rank=row.get("radio_rank"),
                        detections=row["detections"], label_counts=row.get("label_counts"),
                        body_pose=frame.get("body_pose"), camera_pose=frame.get("camera_pose"),
                        relative_navigation_point=relative_place(frame, points), image=str(path.resolve()),
                        count_verification="YOLO detection count only; not guaranteed exact physical count."))
                report["status"] = "candidates_for_review" if chosen else "no_yolo_supported_candidate"
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "raw_calls"}, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intent", choices=["current", "person", "four-chairs", "white-vase-plant"])
    parser.add_argument("--manifest", type=Path, default=Path("data/demo_walk02/active_demo.json"))
    parser.add_argument("--url", default="http://192.168.1.4:8765/mcp")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, choices=range(1, 6), default=3)
    asyncio.run(query(parser.parse_args()))


if __name__ == "__main__":
    main()
