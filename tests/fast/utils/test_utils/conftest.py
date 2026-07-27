from __future__ import annotations

import http.client
import urllib.parse
from collections.abc import Callable, Iterator

import pytest

from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer


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
