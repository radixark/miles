"""HTTP smoke tests for MultiLoRAHTTPServer with a mock upstream (no Ray, no SGLang)."""

import asyncio
import json

import aiohttp
from aiohttp import web
import pytest

from miles.utils.multi_lora import (
    MultiLoRAControllerLogic,
    MultiLoRAHTTPServer,
    make_rid,
)


def _is_dummy(body: dict) -> bool:
    return body.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort"


async def _start_mock_upstream(delay: float = 0.0):
    async def handler(request):
        if delay:
            await asyncio.sleep(delay)
        body = await request.read()
        rid = json.loads(body).get("rid") if body else None
        return web.json_response({"text": "upstream-ok", "rid": rid})

    app = web.Application()
    app.router.add_resource("/{tail:.*}").add_route("*", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _post(session, url, payload):
    async with session.post(url, json=payload) as resp:
        return resp.status, await resp.json()


@pytest.mark.asyncio
async def test_forward_active_returns_upstream():
    upstream_runner, upstream_url = await _start_mock_upstream()
    logic = MultiLoRAControllerLogic(max_adapters=4)
    srv = MultiLoRAHTTPServer(logic, upstream_url)
    await srv.start()
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"http://127.0.0.1:{srv.actual_port}/register_adapter", json={"name": "A"})
            rid = make_rid("A")
            status, body = await _post(s, f"http://127.0.0.1:{srv.actual_port}/generate", {"rid": rid, "text": "hi"})
            assert status == 200
            assert body["text"] == "upstream-ok"
    finally:
        await srv.stop()
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_deregister_mid_flight_dummies():
    upstream_runner, upstream_url = await _start_mock_upstream(delay=0.2)
    logic = MultiLoRAControllerLogic(max_adapters=4)
    srv = MultiLoRAHTTPServer(logic, upstream_url)
    await srv.start()
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"http://127.0.0.1:{srv.actual_port}/register_adapter", json={"name": "A"})
            rid = make_rid("A")
            task = asyncio.create_task(
                _post(s, f"http://127.0.0.1:{srv.actual_port}/generate", {"rid": rid, "text": "hi"})
            )
            await asyncio.sleep(0.05)  # let it be forwarded/in-flight
            await s.post(f"http://127.0.0.1:{srv.actual_port}/deregister_adapter", json={"name": "A"})
            status, body = await task
            assert _is_dummy(body)
            assert body["text"] == ""
    finally:
        await srv.stop()
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_block_retired_adapter():
    upstream_runner, upstream_url = await _start_mock_upstream()
    logic = MultiLoRAControllerLogic(max_adapters=4)
    srv = MultiLoRAHTTPServer(logic, upstream_url)
    await srv.start()
    try:
        async with aiohttp.ClientSession() as s:
            rid = make_rid("A")  # never registered
            status, body = await _post(s, f"http://127.0.0.1:{srv.actual_port}/generate", {"rid": rid, "text": "hi"})
            assert _is_dummy(body)  # blocked, not forwarded
    finally:
        await srv.stop()
        await upstream_runner.cleanup()
