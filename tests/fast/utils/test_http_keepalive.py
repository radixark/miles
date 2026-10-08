"""Real HTTP regression coverage for client stalls across server keep-alive expiry."""

import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from httpcore._async.http11 import AsyncHTTP11Connection

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import http_utils
from miles.utils.http_utils import post, post_bytes_no_retry


@pytest.fixture
def control_http_server():
    received = []

    async def app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        received.append((scope["client"], json.loads(body) if body else {}))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok": true}'})

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, loop="asyncio", http="h11", lifespan="off", timeout_keep_alive=1, log_level="error")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline, "HTTP server did not start"
                time.sleep(0.01)
            yield f"http://127.0.0.1:{port}", received
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "HTTP server did not stop"


@pytest.mark.parametrize("operation", ["sglang", "samples", "create", "delete"])
@pytest.mark.parametrize("response_delay,send_delay", [(0, 1.2), (0.65, 0.5)])
async def test_control_request_survives_client_stall(
    monkeypatch, control_http_server, response_delay, send_delay, operation
):
    url, received = control_http_server
    send_headers = AsyncHTTP11Connection._send_request_headers
    response_closed = AsyncHTTP11Connection._response_closed
    request_number = 0
    response_number = 0

    async def stalled_send(connection, request):
        nonlocal request_number
        request_number += 1
        if request_number == 2:
            # The server runs on a separate thread while this client's event loop is blocked.
            time.sleep(send_delay)
        await send_headers(connection, request)

    async def stalled_response_closed(connection):
        nonlocal response_number
        response_number += 1
        if response_number == 1:
            time.sleep(response_delay)
        await response_closed(connection)

    monkeypatch.setattr(AsyncHTTP11Connection, "_send_request_headers", stalled_send)
    monkeypatch.setattr(AsyncHTTP11Connection, "_response_closed", stalled_response_closed)

    async def request(tag):
        if operation == "sglang":
            return await SGLangApiClient(url).resume_memory_occupation(tags=[tag])
        if operation == "samples":
            result = await post_bytes_no_retry(f"{url}/sessions/id/samples", {"tags": [tag]}, timeout=10)
            return json.loads(result)
        return await post(
            f"{url}/sessions",
            {"tags": [tag]} if operation == "create" else {},
            action="post" if operation == "create" else "delete",
            max_retries=1,
            reuse_connections=False,
        )

    assert await request("weights") == {"ok": True}
    await asyncio.sleep(0.05)
    assert await request("kv_cache") == {"ok": True}

    expected = [{}, {}] if operation == "delete" else [{"tags": ["weights"]}, {"tags": ["kv_cache"]}]
    assert [body for _, body in received] == expected
    assert received[0][0] != received[1][0]
    assert request_number == 2


async def test_default_requests_still_reuse_connections(monkeypatch, control_http_server):
    url, received = control_http_server
    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)
    for _ in range(2):
        assert await post(url, {}, max_retries=1) == {"ok": True}
    assert received[0][0] == received[1][0]


async def test_control_requests_bypass_distributed_generation_pools(monkeypatch, control_http_server):
    url, received = control_http_server
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [object()])

    def unexpected_actor():
        pytest.fail("control call dispatched to a generation actor")

    monkeypatch.setattr(http_utils, "_next_actor", unexpected_actor)
    for action in ("post", "delete"):
        assert await post(url, {}, action=action, reuse_connections=False, max_retries=1) == {"ok": True}
    assert len(received) == 2
    assert received[0][0] != received[1][0]
