"""The MCP server, exercised by a real MCP client over stdio.

Talks to a subprocess exactly as a harness would, so it covers the transport, the
tool schemas and the JSON round trip -- not just the functions underneath.

The text-query test needs the model, so it is marked slow.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import pytest

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from slamibot_fieldmind.paths import PROJECT_ROOT
from slamibot_fieldmind.store import RecordStore

RECORDS = PROJECT_ROOT / "data" / "records"
COLLECTION = "radio_clip_v1"

pytestmark = pytest.mark.skipif(
    not (RECORDS / "records.sqlite").is_file(),
    reason=f"no recorded collection at {RECORDS}; run scripts/record_live.py first",
)


def data_window() -> dict[str, str]:
    """An explicit window covering the recorded data.

    The tools default to the last 24 hours (spec section 7), so a test that leans
    on the default silently starts failing once the recording is a day old --
    which is exactly what happened.
    """
    with RecordStore.open_existing(RECORDS, COLLECTION) as store:
        first, last = store.time_range()
    assert first is not None, "the collection is empty"
    return {
        "start_time": (first - dt.timedelta(minutes=1)).isoformat(),
        "end_time": (last + dt.timedelta(minutes=1)).isoformat(),
    }


def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "slamibot_fieldmind.mcp_server", "--root", str(RECORDS), "--collection-id", COLLECTION],
    )


async def with_session(call) -> object:
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await call(session)


def run(call) -> object:
    import anyio

    return anyio.run(with_session, call)


def test_server_advertises_the_expected_tools():
    async def call(session):
        return await session.list_tools()

    tools = run(call).tools
    names = sorted(tool.name for tool in tools)
    assert names == ["compare_frames", "describe_view", "find_frames", "get_frame", "index_status", "search_frames"]
    assert "get_frame_image" not in names, "frames are never stored, so no image tool may exist"


def test_tool_descriptions_carry_the_spec_caveats():
    """Spec section 9: the harness reads the description, not our docs."""

    async def call(session):
        return await session.list_tools()

    by_name = {tool.name: (tool.description or "") for tool in run(call).tools}
    search = by_name["search_frames"].lower()
    assert "not a probability" in search
    assert "candidate" in search
    assert "stale" in by_name["index_status"].lower()
    assert "not proof" in by_name["find_frames"].lower()
    assert "stand-in" in by_name["compare_frames"].lower()


def test_index_status_reports_coverage_without_loading_the_model():
    async def call(session):
        result = await session.call_tool("index_status", {})
        return result

    payload = json.loads(run(call).content[0].text)
    assert payload["status"] == "ok"
    assert payload["indexed_frames"] > 0
    assert payload["stores_images"] is False
    assert payload["threshold_calibrated"] is False
    # This integration fixture is a real capture directory and may contain a
    # damaged historical partition. The server must report that damage honestly.
    with RecordStore.open_existing(RECORDS, COLLECTION) as store:
        expected_integrity = store.verify()
    assert payload["vector_partitions_ok"] is expected_integrity["ok"]
    assert payload["integrity_problems"] == expected_integrity["orphan_or_missing"]
    assert payload["pose_source"] == "simulated"


def test_find_frames_by_label():
    async def call(session):
        return await session.call_tool("find_frames", {"labels": ["tv"], "limit": 2, **data_window()})

    payload = json.loads(run(call).content[0].text)
    assert payload["status"] == "ok"
    assert payload["matched_frame_count"] > 0
    assert payload["returned"] <= 2
    assert "tv" in payload["frames"][0]["label_counts"]


def test_find_frames_reports_an_absent_label_without_implying_absence():
    """The window has frames; the label is what excluded them. The reply must say
    so, not blame the window."""

    async def call(session):
        return await session.call_tool("find_frames", {"labels": ["fire truck"], **data_window()})

    payload = json.loads(run(call).content[0].text)
    assert payload["status"] == "no_frames_in_range"
    assert payload["empty_because"] == "filters"
    assert payload["frames_in_window_before_filters"] > 0
    assert "filter is what excluded them" in payload["note"]
    assert "not proof" in payload["caveat"]


def test_get_and_compare_round_trip():
    async def call(session):
        found = await session.call_tool("find_frames", {"labels": ["tv"], "limit": 3, **data_window()})
        ids = [f["frame_id"] for f in json.loads(found.content[0].text)["frames"]]
        single = await session.call_tool("get_frame", {"frame_id": ids[0]})
        compared = await session.call_tool("compare_frames", {"frame_ids": ids[:2]})
        return json.loads(single.content[0].text), json.loads(compared.content[0].text)

    single, compared = run(call)
    assert single["status"] == "ok"
    assert single["vector"]["dim"] == 1024
    assert "no image" in single["note"].lower()
    assert compared["status"] == "ok"
    assert len(compared["frames"]) == 2
    assert compared["frames"][0]["seconds_from_first"] == 0.0
    assert compared["pose_source"] == "simulated"


def test_get_frame_on_an_unknown_id_is_not_an_error():
    async def call(session):
        return await session.call_tool("get_frame", {"frame_id": "camera_01-nope"})

    payload = json.loads(run(call).content[0].text)
    assert payload["status"] == "not_found"


@pytest.mark.slow
def test_search_frames_loads_the_model_and_ranks():
    async def call(session):
        return await session.call_tool(
            "search_frames", {"query": "a computer monitor", "top_k": 3, **data_window()}
        )

    payload = json.loads(run(call).content[0].text)
    assert payload["status"] == "ok"
    assert payload["threshold_calibrated"] is False
    assert payload["indexed_frame_count_in_range"] > 0
    assert len(payload["matches"]) <= 3
    first = payload["matches"][0]
    assert first["similarity"] > 0.0
    assert "group_id" in first
    assert "not an event boundary" in payload["grouping_note"]


@pytest.mark.slow
def test_search_frames_respects_a_narrow_window():
    """A window with no records must say so, not return something from outside it."""

    async def call(session):
        return await session.call_tool(
            "search_frames",
            {"query": "a computer monitor", "start_time": "2000-01-01T00:00:00Z", "end_time": "2000-01-02T00:00:00Z"},
        )

    payload = json.loads(run(call).content[0].text)
    assert payload["status"] == "no_frames_in_range"
    assert payload["matches"] == []
    assert payload["empty_because"] == "time_window"
    assert "not evidence that nothing was seen" in payload["note"]
    assert payload["collection_total_frames"] > 0, "the note should point at where the data actually is"


def test_an_empty_window_explains_itself_rather_than_implying_absence():
    """The default window is the last 24 hours. On aged data it is empty, and the
    reply must point at the window instead of letting a harness conclude that
    nothing was seen."""

    async def call(session):
        return await session.call_tool("find_frames", {"labels": ["tv"]})

    payload = json.loads(run(call).content[0].text)
    if payload["status"] == "ok":
        pytest.skip("the recording is recent enough that the default window still covers it")
    assert payload["collection_is_empty"] is False
    assert payload["collection_total_frames"] > 0
    assert payload["hours_since_latest_record"] > 24
    assert "not evidence" in payload["note"]
    assert "Widen start_time" in payload["note"]


def test_the_declared_window_matches_the_data():
    """Guards the guard: if data_window() were wrong the tests above would pass
    for the wrong reason."""

    async def call(session):
        return await session.call_tool("find_frames", {"labels": ["tv"], "limit": 1, **data_window()})

    payload = json.loads(run(call).content[0].text)
    assert payload["status"] == "ok", payload
    assert payload["matched_frame_count"] > 0
