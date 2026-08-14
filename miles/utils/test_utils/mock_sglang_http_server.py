from __future__ import annotations

import dataclasses
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    payload: dict[str, Any] | None


class MockSGLangHttpServer:
    def __init__(self, response_payload: dict[str, Any] | None = None):
        self._response_payload = response_payload if response_payload is not None else {"mock": True}
        self._requests: list[RecordedRequest] = []
        self._lock = threading.Lock()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def host(self) -> str:
        return self._server.server_address[0]

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
        self._thread.join(timeout=5)

    def _record(self, method: str, path: str, payload: dict[str, Any] | None) -> None:
        with self._lock:
            self._requests.append(RecordedRequest(method=method, path=path.split("?")[0], payload=payload))

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

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


class MockSGLangHttpServerPool:
    def __init__(self):
        self._servers: dict[int, MockSGLangHttpServer] = {}
        self._lock = threading.Lock()

    def for_rank(self, rank: int) -> MockSGLangHttpServer:
        with self._lock:
            if rank not in self._servers:
                self._servers[rank] = MockSGLangHttpServer()
            return self._servers[rank]

    def new_for_rank(self, rank: int) -> MockSGLangHttpServer:
        with self._lock:
            previous = self._servers.pop(rank, None)
            server = MockSGLangHttpServer()
            self._servers[rank] = server
        if previous is not None:
            previous.close()
        return server

    def close(self) -> None:
        with self._lock:
            servers = list(self._servers.values())
            self._servers.clear()
        for server in servers:
            server.close()
