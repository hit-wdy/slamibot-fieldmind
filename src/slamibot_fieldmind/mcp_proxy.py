"""Stdio MCP proxy to an existing backend; never loads a model or camera."""

import argparse
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def run(url):
    async with streamable_http_client(
        url, max_sse_event_size=16 * 1024 * 1024
    ) as streams:
        async with ClientSession(streams[0], streams[1]) as remote:
            await remote.initialize()

            async def list_tools(context, params):
                return await remote.list_tools(params=params)

            async def call_tool(context, params):
                return await remote.call_tool(params.name, params.arguments or {})

            server = Server(
                "slamibot-fieldmind-proxy",
                on_list_tools=list_tools,
                on_call_tool=call_tool,
            )

            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    asyncio.run(run(parser.parse_args().url))


if __name__ == "__main__":
    main()
