from __future__ import annotations

import http.client
import json
import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer


class RecordingSessionBackend:
    def __init__(self) -> None:
        self.cpu_commands: list[str] = []
        self.train_launches: list[dict[str, Any]] = []

    def exec_command_cpu(self, command: str) -> None:
        self.cpu_commands.append(command)

    def execute_train(self, **kwargs: Any) -> None:
        self.train_launches.append(kwargs)
        metrics_path = Path(kwargs["extra_env_vars"]["MILES_SESSION_VERIFY_METRICS_PATH"])
        metrics_path.write_text(json.dumps({"driver_events": ["append_tool"], "had_assistant_mismatch": False}) + "\n")


class RecordingSessionConfig:
    def __init__(self, backend: RecordingSessionBackend) -> None:
        self.backend = backend
        self.create_backend_calls = 0

    def create_backend(self) -> RecordingSessionBackend:
        self.create_backend_calls += 1
        return self.backend


@pytest.fixture
def make_server() -> Iterator[Callable[..., MockSGLangHttpServer]]:
    servers: list[MockSGLangHttpServer] = []

    def _make(**kwargs) -> MockSGLangHttpServer:
        server = MockSGLangHttpServer(**kwargs)
        servers.append(server)
        return server

    yield _make

    for server in servers:
        server.close()


def connect_via_url(server: MockSGLangHttpServer) -> http.client.HTTPConnection:
    parts = urllib.parse.urlsplit(server.url)
    assert parts.scheme == "http"
    return http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
