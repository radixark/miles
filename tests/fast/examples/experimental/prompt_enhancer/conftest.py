import base64
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


class FakeServer:
    """An in-process HTTP server answering POSTs on one path; `status` sets the reply code for every request."""

    def __init__(self, path: str, answer) -> None:
        self.requests: list[dict] = []
        self.status = 200
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                fake.requests.append(body)
                status = fake.status if self.path == path else 404
                payload = json.dumps(answer(body) if status == 200 else {"detail": "refused"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.host, self.port = self._server.server_address
        self.url = f"http://{self.host}:{self.port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def engine():
    """An SGLang Diffusion stand-in: one PNG of the requested size, as `b64_json`."""
    server = FakeServer(
        "/v1/images/generations",
        lambda body: {"data": [{"b64_json": base64.b64encode(png(body["width"], body["height"])).decode()}]},
    )
    yield server
    server.close()


@pytest.fixture
def scorer():
    """A scorer_server.py stand-in that scores every image 0.8."""
    server = FakeServer("/score", lambda body: {"image_reward": 0.8})
    yield server
    server.close()
