from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import server as session_server_module
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.server import SessionServer, main
from miles.utils.workers.argv_utils import config_to_argv


class TestSessionServer:
    def test_constructor_applies_configured_timeout_to_proxy_client(self):
        """The configured timeout reaches the shared proxy client instead of an httpx default."""
        server = SessionServer(make_session_server_config(timeout=7.5))

        assert server.client.timeout == httpx.Timeout(7.5)

    @pytest.mark.asyncio
    async def test_request_hook_can_add_policy_and_retry_admission_rejection(self, monkeypatch):
        async def hook(hook_args, context, request):
            assert hook_args == {"minimum_version": 3}
            request["payload"]["weight_version"] = {"min_version": hook_args["minimum_version"]}
            request["headers"]["X-Session-ID"] = context.session_id
            request["max_attempts"] = 2
            request["retry_interval"] = 0

        monkeypatch.setattr(session_server_module, "load_function", lambda _path: hook)
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(409 if len(requests) == 1 else 200, json={"ok": True})

        server = SessionServer(
            make_session_server_config(
                custom_rollout_request_hook_path="fake.request_hook",
                custom_rollout_request_hook_args={"minimum_version": 3},
            )
        )
        await server.client.aclose()
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await server.do_proxy(
                ProxyRequest(method="POST", session_id="session-1"),
                "v1/chat/completions",
                body=b'{"messages":[]}',
                headers={"content-type": "application/json"},
            )
        finally:
            await server.client.aclose()

        assert result["status_code"] == 200
        assert len(requests) == 2
        assert json.loads(requests[1].content)["weight_version"] == {"min_version": 3}
        assert requests[1].headers["X-Session-ID"] == "session-1"

    @pytest.mark.asyncio
    async def test_request_hook_does_not_retry_ambiguous_server_error(self, monkeypatch):
        def hook(_hook_args, _context, request):
            request["max_attempts"] = 3
            request["retry_interval"] = 0

        monkeypatch.setattr(session_server_module, "load_function", lambda _path: hook)
        request_count = 0

        async def handler(_request):
            nonlocal request_count
            request_count += 1
            return httpx.Response(500, json={"error": "failed after dispatch"})

        server = SessionServer(make_session_server_config(custom_rollout_request_hook_path="fake.request_hook"))
        await server.client.aclose()
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await server.do_proxy(
                ProxyRequest(method="POST", session_id="session-1"),
                "v1/chat/completions",
                body=b"{}",
                headers={"content-type": "application/json"},
            )
        finally:
            await server.client.aclose()

        assert result["status_code"] == 500
        assert request_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error_type", "expected_attempts"),
        [(httpx.ConnectError, 2), (httpx.ReadError, 1)],
    )
    async def test_request_hook_retries_only_pre_dispatch_transport_errors(
        self, monkeypatch, error_type, expected_attempts
    ):
        def hook(_hook_args, _context, request):
            request["max_attempts"] = 2
            request["retry_interval"] = 0

        monkeypatch.setattr(session_server_module, "load_function", lambda _path: hook)
        request_count = 0

        async def send_request(*_args, **_kwargs):
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                raise error_type("transport failed")
            return httpx.Response(200, json={"ok": True})

        server = SessionServer(make_session_server_config(custom_rollout_request_hook_path="fake.request_hook"))
        await server.client.aclose()
        server.client = SimpleNamespace(request=send_request)

        result = await server.do_proxy(
            ProxyRequest(method="POST", session_id="session-1"),
            "v1/chat/completions",
            body=b"{}",
            headers={"content-type": "application/json"},
        )

        assert result["status_code"] == (200 if error_type is httpx.ConnectError else 502)
        assert request_count == expected_attempts


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
