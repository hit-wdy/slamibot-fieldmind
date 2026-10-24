"""Call every MCP tool through a real MCP client and print what comes back.

This is the one-command way to see the tool surface work. It spawns the server
over stdio exactly as a harness would, so if this passes, the transport, the tool
schemas and the JSON round trip are all fine.

    python scripts/mcp_demo.py                       # everything (loads the model)
    python scripts/mcp_demo.py --skip-model          # metadata tools only, fast
    python scripts/mcp_demo.py --query "a laptop"    # your own phrase
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

import datetime as dt

from slamibot_fieldmind.paths import PROJECT_ROOT
from slamibot_fieldmind.store import RecordStore

DEFAULT_ROOT = PROJECT_ROOT / "data" / "records"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--collection-id", default="radio_clip_v1")
    parser.add_argument("--query", default="a computer monitor")
    parser.add_argument("--skip-model", action="store_true", help="skip the tools that need RADIO")
    return parser.parse_args(argv)


def show(title: str, payload: dict, drop: tuple[str, ...] = ()) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps({k: v for k, v in payload.items() if k not in drop}, indent=2, ensure_ascii=False))
    for key in drop:
        value = payload.get(key)
        if value is None:
            continue
        rows = value if isinstance(value, list) else [value]
        for row in rows[:3]:
            print("   ", json.dumps(row, ensure_ascii=False)[:240])


async def run(args: argparse.Namespace) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "slamibot_fieldmind.mcp_server", "--root", str(args.root),
              "--collection-id", args.collection_id],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print(f"server exposes {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}")

            async def call(name: str, arguments: dict) -> dict:
                result = await session.call_tool(name, arguments)
                if result.is_error:
                    raise RuntimeError(f"{name} failed: {result.content[0].text[:400]}")
                return json.loads(result.content[0].text)

            status = await call("index_status", {})
            show("index_status", status, drop=("cameras",))
            print("   cameras:", json.dumps(status.get("cameras"), ensure_ascii=False))

            # The tools default to the last 24 hours. Spelled out here so the demo
            # still finds a recording that is older than that.
            window = _data_window(args.root, args.collection_id)
            print(f"   using the collection's own range: {window['start_time']} .. {window['end_time']}")

            label = (status.get("detection_labels") or {"tv": 1})
            busiest = max(label, key=lambda k: label[k])
            found = await call("find_frames", {"labels": [busiest], "limit": 3, **window})
            show(f"find_frames(labels=['{busiest}'])", found, drop=("frames",))

            absent = await call("find_frames", {"labels": ["fire truck"], **window})
            print(f"\n=== find_frames(labels=['fire truck']) ===\n"
                  f"    status={absent['status']} matched={absent['matched_frame_count']}")
            print(f"    caveat: {absent['caveat']}")

            frame_ids = [f["frame_id"] for f in found["frames"][:2]]
            if len(frame_ids) == 2:
                compared = await call("compare_frames", {"frame_ids": frame_ids})
                show("compare_frames", compared, drop=("frames",))
                for row in compared["frames"]:
                    print("   ", json.dumps(row, ensure_ascii=False)[:240])

            if frame_ids:
                single = await call("get_frame", {"frame_id": frame_ids[0]})
                show("get_frame", single, drop=("detections",))

            if args.skip_model:
                print("\n(skipped search_frames: it needs RADIO)")
                return 0

            print("\n... loading RADIO for search_frames, this takes ~20 s")
            searched = await call("search_frames", {"query": args.query, "top_k": 3, **window})
            show("search_frames", searched, drop=("matches",))
            print(f"    threshold_calibrated={searched['threshold_calibrated']}")
            print(f"    {searched.get('calibration_note', '')}")
            for match in searched["matches"]:
                print("   ", json.dumps(match, ensure_ascii=False)[:240])

            empty = await call("search_frames", {
                "query": args.query,
                "start_time": "2000-01-01T00:00:00Z",
                "end_time": "2000-01-02T00:00:00Z",
            })
            print(f"\n=== search_frames over an empty window ===")
            print(f"    status={empty['status']} matches={len(empty['matches'])}")
            print(f"    note: {empty.get('note')}")
    return 0


def _data_window(root: Path, collection_id: str) -> dict[str, str]:
    with RecordStore.open_existing(root, collection_id) as store:
        first, last = store.time_range()
    assert first is not None, "the collection is empty"
    return {
        "start_time": (first - dt.timedelta(minutes=1)).isoformat(),
        "end_time": (last + dt.timedelta(minutes=1)).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.root / "records.sqlite").is_file():
        print(f"no records at {args.root}. Run scripts/record_live.py first.", file=sys.stderr)
        return 2
    return anyio.run(run, args)


if __name__ == "__main__":
    raise SystemExit(main())
