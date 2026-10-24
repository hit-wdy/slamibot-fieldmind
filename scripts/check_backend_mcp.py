"""Exercise all nine backend tools through the stdio proxy; save evidence/report."""

import argparse
import asyncio
import base64
import datetime as dt
import json
from pathlib import Path
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def run(args):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "slamibot_fieldmind.mcp_proxy", "--url", args.url],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            if len(names) != 9:
                raise RuntimeError("expected nine backend tools: " + repr(names))

            async def call(name, **arguments):
                response = await session.call_tool(name, arguments)
                if response.is_error:
                    raise RuntimeError(f"{name}: {response.content}")
                return json.loads(response.content[0].text)

            report = dict(
                tools=names,
                index_status=await call("index_status"),
                describe_view=await call("describe_view", **({"camera_id": args.camera_id} if args.camera_id else {})),
                list_sessions=await call("list_sessions"),
                list_places=await call("list_places"),
            )
            sessions = [s for s in report["list_sessions"]["sessions"] if s["frames"]]
            if args.session_id:
                sessions = [s for s in sessions if s["session_id"] == args.session_id]
            if not sessions:
                raise RuntimeError("no session with observations to validate")
            latest = sessions[0]
            start = (
                dt.datetime.fromisoformat(latest["first_observation"])
                - dt.timedelta(seconds=1)
            ).isoformat()
            end = (
                dt.datetime.fromisoformat(latest["last_observation"])
                + dt.timedelta(seconds=1)
            ).isoformat()
            filters = dict(
                session_id=latest["session_id"], start_time=start, end_time=end
            )
            report["find_frames"] = await call("find_frames", limit=2, **filters)
            report["person"] = await call("search_frames", query="a person", **filters)
            report["whale"] = await call(
                "search_frames", query="a large whale", **filters
            )
            frames = report["find_frames"]["frames"]
            if not frames:
                raise RuntimeError("session has no queryable frames")
            frame_id = frames[0]["frame_id"]
            report["get_frame"] = await call("get_frame", frame_id=frame_id)
            if len(frames) >= 2:
                report["compare_frames"] = await call(
                    "compare_frames", frame_ids=[f["frame_id"] for f in frames]
                )
            response = await session.call_tool(
                "get_frame_image", dict(frame_id=frame_id, overlay=True)
            )
            if response.is_error or response.content[0].type != "image":
                raise RuntimeError("image evidence was not returned")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            evidence = args.out.with_suffix(".jpg")
            evidence.write_bytes(base64.b64decode(response.content[0].data))
            report["image_evidence"] = str(evidence.resolve())
            args.out.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                json.dumps(
                    dict(
                        report=str(args.out.resolve()),
                        tool_count=len(names),
                        states=report["index_status"]["states"],
                        current_status=report["describe_view"]["status"],
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session-id", help="Limit validation to a specific replay session")
    parser.add_argument("--camera-id", help="Limit current-view inspection to a logical camera")
    asyncio.run(run(parser.parse_args()))
