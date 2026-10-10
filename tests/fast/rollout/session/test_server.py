from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import server as session_server_module
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.server import SessionServer, main
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.workers.argv_utils import config_to_argv


class TestSessionServer:
    def test_proxy_client_lifecycle_without_requests(self):
        """Uvicorn creates the configured client at startup and closes it without needing a request."""
        server = SessionServer(make_session_server_config(timeout=7.5))
        http_server = UvicornThreadServer(server.app, host="127.0.0.1", port=find_available_port(32100))
        http_server.start()
        try:
            client = server.client
            assert not client.closed
            assert client.timeout == aiohttp.ClientTimeout(total=None, connect=7.5, sock_read=7.5)
        finally:
            http_server.stop()
        assert client.closed


@pytest.fixture
def echo_backend():
    """An upstream that echoes the request it received, answering 201 with an extra header."""
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        received = {"headers": dict(request.headers), "body": (await request.body()).decode()}
        return JSONResponse(received, status_code=201, headers={"X-Upstream": "yes"})

    @app.post("/moved")
    async def moved():
        return Response(status_code=307, headers={"location": "/echo"})

    server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(32100))
    server.start()
    try:
        yield server.url
    finally:
        server.stop()


async def _proxy(server: SessionServer, path: str, *, body: bytes = b"{}", headers: dict | None = None) -> dict:
    async with server.app.router.lifespan_context(server.app):
        return await server.do_proxy(ProxyRequest(method="POST"), path, body=body, headers=headers or {})


def _proxy_to(backend_url: str, path: str, **kwargs) -> dict:
    return asyncio.run(_proxy(SessionServer(make_session_server_config(backend_url=backend_url)), path, **kwargs))


class TestDoProxy:
    def test_forwards_body_and_headers_without_adding_a_content_type(self, echo_backend):
        """SGLang parses a body sent without Content-Type as JSON, so the proxy must not invent one."""
        body = b'{"a": 1}'
        result = _proxy_to(echo_backend, "echo", body=body, headers={"x-trace": "abc", "content-length": "999"})

        received = json.loads(result["response_body"])
        assert received["body"] == body.decode()
        assert received["headers"]["x-trace"] == "abc"
        assert "content-type" not in received["headers"]
        assert received["headers"]["content-length"] == str(len(body))

    def test_returns_the_upstream_status_and_lower_case_headers(self, echo_backend):
        result = _proxy_to(echo_backend, "echo")

        assert result["status_code"] == 201
        assert result["headers"]["x-upstream"] == "yes"
        assert result["headers"]["content-type"] == "application/json"

    def test_does_not_follow_redirects(self, echo_backend):
        assert _proxy_to(echo_backend, "moved")["status_code"] == 307

    def test_unreachable_backend_is_a_502(self):
        result = _proxy_to(f"http://127.0.0.1:{find_available_port(33100)}", "echo")

        assert result["status_code"] == 502
        assert json.loads(result["response_body"])["error"].startswith("backend transport error: ")

    def test_reply_cut_off_mid_body_is_a_502(self):
        """A failed body read is a transport error too, as when httpx read the body inside request()."""

        async def run():
            async def truncated_reply(reader, writer):
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial")
                await writer.drain()
                writer.close()

            upstream = await asyncio.start_server(truncated_reply, "127.0.0.1", 0)
            port = upstream.sockets[0].getsockname()[1]
            try:
                server = SessionServer(make_session_server_config(backend_url=f"http://127.0.0.1:{port}"))
                return await _proxy(server, "x")
            finally:
                upstream.close()

        assert asyncio.run(run())["status_code"] == 502

    def test_backend_that_stops_reading_the_upload_is_a_502(self):
        async def run():
            transports = []

            class NonReadingBackend(asyncio.Protocol):
                def connection_made(self, transport):
                    transports.append(transport)
                    transport.pause_reading()

            upstream = await asyncio.get_running_loop().create_server(NonReadingBackend, "127.0.0.1", 0)
            port = upstream.sockets[0].getsockname()[1]
            server = SessionServer(make_session_server_config(backend_url=f"http://127.0.0.1:{port}", timeout=0.1))
            try:
                return await asyncio.wait_for(_proxy(server, "x", body=b"x" * (16 * 1024 * 1024)), timeout=2)
            finally:
                upstream.close()
                for transport in transports:
                    transport.abort()
                await upstream.wait_closed()

        assert asyncio.run(run())["status_code"] == 502

    def test_reply_can_exceed_one_timeout_while_each_read_makes_progress(self):
        async def run():
            tasks = set()

            async def reply(reader, writer):
                tasks.add(asyncio.current_task())
                try:
                    await reader.readuntil(b"\r\n\r\n")
                    await reader.readexactly(2)
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n")
                    for _ in range(6):
                        await asyncio.sleep(0.05)
                        writer.write(b"x")
                        await writer.drain()
                finally:
                    writer.close()
                    await writer.wait_closed()
                    tasks.remove(asyncio.current_task())

            upstream = await asyncio.start_server(reply, "127.0.0.1", 0)
            port = upstream.sockets[0].getsockname()[1]
            server = SessionServer(make_session_server_config(backend_url=f"http://127.0.0.1:{port}", timeout=0.2))
            try:
                return await asyncio.wait_for(_proxy(server, "x"), timeout=2)
            finally:
                upstream.close()
                await upstream.wait_closed()
                await asyncio.gather(*tasks)

        result = asyncio.run(run())
        assert result["status_code"] == 200
        assert result["response_body"] == b"xxxxxx"


def test_run_session_server_suppresses_routine_request_logs(monkeypatch):
    app = object()
    uvicorn_call = {}
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    monkeypatch.setattr(httpx_logger, "level", logging.NOTSET)
    monkeypatch.setattr(httpcore_logger, "level", logging.NOTSET)

    class FakeSessionServer:
        def __init__(self, config):
            self.app = app

    monkeypatch.setattr(session_server_module, "configure_logger_raw", lambda *_: None)
    monkeypatch.setattr(session_server_module.setproctitle, "setproctitle", lambda *_: None)
    monkeypatch.setattr(session_server_module, "SessionServer", FakeSessionServer)

    def fake_uvicorn_run(received_app, **kwargs):
        uvicorn_call["app"] = received_app
        uvicorn_call.update(kwargs)

    monkeypatch.setattr(session_server_module.uvicorn, "run", fake_uvicorn_run)
    config = make_session_server_config(host="127.0.0.1", port=31001)

    session_server_module.run_session_server(config)

    assert httpx_logger.level == logging.WARNING
    assert httpcore_logger.level == logging.WARNING
    assert uvicorn_call == {
        "app": app,
        "host": "127.0.0.1",
        "port": 31001,
        "log_level": "info",
        "access_log": False,
    }


class TestMain:
    def test_feeds_the_parsed_config_to_the_server(self, monkeypatch):
        """The CLI parses the config payload losslessly."""
        calls = []
        monkeypatch.setattr(session_server_module, "run_session_server", lambda config: calls.append(config))
        config = make_session_server_config(port=5005, instance_id="abc", backend_url="http://127.0.0.1:3000")

        main(config_to_argv(config))

        assert calls == [config]

    def test_missing_config_is_rejected(self):
        """The config payload is mandatory for a session server."""
        with pytest.raises(SystemExit):
            main([])
