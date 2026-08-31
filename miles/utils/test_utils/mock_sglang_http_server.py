from __future__ import annotations

import dataclasses
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    payload: dict[str, Any] | None


class MockSGLangHttpServer:
    def __init__(self, response_payload: dict[str, Any] | None = None, port: int = 0):
        self._response_payload = response_payload if response_payload is not None else {"mock": True}
        self._requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._connections: set[socket.socket] = set()

        self._server = ThreadingHTTPServer(("0.0.0.0", port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def host(self) -> str:
        return get_current_node_ip()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self._requests)

    @property
    def paths(self) -> list[str]:
        return [request.path for request in self.requests]

    def payloads_of(self, path: str) -> list[dict[str, Any] | None]:
        return [request.payload for request in self.requests if request.path == path]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        for connection in self._drain_connections():
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                logger.debug("mock sglang http server: connection was already down", exc_info=True)
            connection.close()
        self._thread.join(timeout=5)

    def _record(self, method: str, path: str, payload: dict[str, Any] | None) -> None:
        with self._lock:
            self._requests.append(RecordedRequest(method=method, path=path.split("?")[0], payload=payload))

    def _register_connection(self, connection: socket.socket) -> None:
        with self._lock:
            self._connections.add(connection)

    def _unregister_connection(self, connection: socket.socket) -> None:
        with self._lock:
            self._connections.discard(connection)

    def _drain_connections(self) -> list[socket.socket]:
        with self._lock:
            connections = list(self._connections)
            self._connections.clear()
        return connections

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                server._register_connection(self.connection)

            def finish(self):
                server._unregister_connection(self.connection)
                super().finish()

            def do_GET(self):
                server._record("GET", self.path, None)
                self._respond()

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                payload = json.loads(raw) if raw else None
                server._record("POST", self.path, payload)
                self._respond()

            def log_message(self, format, *args):
                logger.debug("mock sglang http server: " + format, *args)

            def _respond(self):
                body = json.dumps(server._response_payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return _Handler
