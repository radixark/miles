from __future__ import annotations

import http.client

import pytest

from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer


class TestClose:
    def test_close_severs_an_already_established_connection(self):
        """A crashed engine must stop serving a client that already holds a keep-alive connection."""
        server = MockSGLangHttpServer()
        connection = http.client.HTTPConnection(server.host, server.port, timeout=5)
        connection.request("GET", "/before")
        assert connection.getresponse().status == 200

        server.close()

        with pytest.raises(OSError):
            connection.request("GET", "/after")
            connection.getresponse().read()
        assert server.paths == ["/before"]

    def test_close_refuses_a_fresh_connection(self):
        """A crashed engine must not accept new connections either."""
        server = MockSGLangHttpServer()
        host, port = server.host, server.port
        server.close()

        connection = http.client.HTTPConnection(host, port, timeout=5)
        with pytest.raises(OSError):
            connection.request("GET", "/after")
