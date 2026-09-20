from __future__ import annotations

import logging

import httpx
import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import server as session_server_module
from miles.rollout.session.server import SessionServer, main
from miles.utils.workers.argv_utils import config_to_argv


class TestSessionServer:
    def test_constructor_applies_configured_timeout_to_proxy_client(self):
        """The configured timeout reaches the shared proxy client instead of an httpx default."""
        server = SessionServer(make_session_server_config(timeout=7.5))

        assert server.client.timeout == httpx.Timeout(7.5)


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
