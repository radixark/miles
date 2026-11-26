from __future__ import annotations

import asyncio
import base64
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.image_generation import (
    ImageGenerationClient,
    ImageGenerationUnavailable,
    ImageRequest,
    image_generation_client,
)
from miles.utils.retry_utils import NonRetryableError


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + b"\x7f\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


@dataclass
class _Reply:
    status: int = 200
    delay_secs: float = 0.0
    size: tuple[int, int] | None = None  # None: the size the request asked for
    body: bytes | None = None


class _FakeEngine:
    """An SGLang Diffusion stand-in answering /health and /v1/images/generations from a script of replies."""

    def __init__(self, replies: list[_Reply] | None = None) -> None:
        self.requests: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._replies = list(replies or [])
        self._lock = threading.Lock()
        engine = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._send(200 if self.path == "/health" else 404, b"{}")

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if self.path != "/v1/images/generations":
                    self._send(404, b"{}")
                    return
                status, payload = engine._answer(body)
                self._send(status, payload)

            def _send(self, status: int, payload: bytes) -> None:
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _answer(self, body: dict) -> tuple[int, bytes]:
        with self._lock:
            self.requests.append(body)
            reply = self._replies.pop(0) if self._replies else _Reply()
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(reply.delay_secs)
            if reply.body is not None:
                return reply.status, reply.body
            if reply.status != 200:
                return reply.status, json.dumps({"detail": f"scripted {reply.status}"}).encode()
            width, height = reply.size or (body["width"], body["height"])
            image = base64.b64encode(_png(width, height)).decode()
            return 200, json.dumps({"id": "x", "data": [{"b64_json": image}]}).encode()
        finally:
            with self._lock:
                self.in_flight -= 1

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def engine_factory():
    engines: list[_FakeEngine] = []

    def make(*replies: _Reply) -> _FakeEngine:
        engines.append(_FakeEngine(list(replies)))
        return engines[-1]

    yield make
    for engine in engines:
        engine.close()


_REQUEST = ImageRequest(prompt="a red cube", seed=1234, width=64, height=32)


def _client(url: str, **kwargs) -> ImageGenerationClient:
    kwargs = {"timeout_secs": 5.0, "max_retries": 2, "retry_initial_delay_secs": 0.01, **kwargs}
    return ImageGenerationClient(url, **kwargs)


class TestRequest:
    def test_posts_one_png_request_and_returns_its_bytes(self, engine_factory):
        engine = engine_factory()
        request = ImageRequest(
            prompt="a red cube", seed=7, width=64, height=32, num_inference_steps=8, guidance_scale=0.0
        )

        png = asyncio.run(_client(engine.url).generate_image(request))

        assert png == _png(64, 32)
        assert engine.requests == [
            {
                "prompt": "a red cube",
                "seed": 7,
                "width": 64,
                "height": 32,
                "num_inference_steps": 8,
                "guidance_scale": 0.0,
                "n": 1,
                "response_format": "b64_json",
                "output_format": "png",
            }
        ]

    def test_unset_optional_fields_are_left_to_the_engine(self, engine_factory):
        engine = engine_factory()

        asyncio.run(_client(engine.url).generate_image(_REQUEST))

        assert set(engine.requests[0]) == {
            "prompt",
            "seed",
            "width",
            "height",
            "n",
            "response_format",
            "output_format",
        }


class TestErrors:
    @pytest.mark.parametrize("status", [400, 404, 408, 422, 429])
    def test_a_4xx_is_not_retried(self, engine_factory, status):
        engine = engine_factory(_Reply(status=status))

        with pytest.raises(NonRetryableError, match=f"{status}.*scripted {status}"):
            asyncio.run(_client(engine.url).generate_image(_REQUEST))
        assert len(engine.requests) == 1

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_a_5xx_is_retried_then_reported_unavailable(self, engine_factory, status):
        engine = engine_factory(*[_Reply(status=status)] * 4)

        with pytest.raises(ImageGenerationUnavailable, match=f"3 attempts failed.*HTTP {status}"):
            asyncio.run(_client(engine.url, max_retries=2).generate_image(_REQUEST))
        assert len(engine.requests) == 3

    def test_a_5xx_then_an_image_succeeds(self, engine_factory):
        engine = engine_factory(_Reply(status=503), _Reply(status=500))

        png = asyncio.run(_client(engine.url, max_retries=2).generate_image(_REQUEST))

        assert png == _png(64, 32)
        assert len(engine.requests) == 3

    def test_a_timeout_is_retried_then_reported_unavailable(self, engine_factory):
        engine = engine_factory(*[_Reply(delay_secs=1.0)] * 2)

        with pytest.raises(ImageGenerationUnavailable, match="2 attempts failed.*no answer within 0.2s"):
            asyncio.run(_client(engine.url, timeout_secs=0.2, max_retries=1).generate_image(_REQUEST))
        assert len(engine.requests) == 2

    def test_no_retries_means_one_attempt(self, engine_factory):
        engine = engine_factory(_Reply(status=500))

        with pytest.raises(ImageGenerationUnavailable, match="1 attempts failed"):
            asyncio.run(_client(engine.url, max_retries=0).generate_image(_REQUEST))
        assert len(engine.requests) == 1

    def test_an_unreachable_engine_is_reported_unavailable(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        with pytest.raises(ImageGenerationUnavailable, match="2 attempts failed"):
            asyncio.run(_client(f"http://127.0.0.1:{port}", max_retries=1).generate_image(_REQUEST))

    def test_a_size_mismatch_is_not_retried(self, engine_factory):
        engine = engine_factory(_Reply(size=(64, 64)))

        with pytest.raises(NonRetryableError, match="asked for a 64x32 image, the engine made 64x64"):
            asyncio.run(_client(engine.url).generate_image(_REQUEST))
        assert len(engine.requests) == 1

    @pytest.mark.parametrize(
        "body, match",
        [
            (b"not json", "not JSON"),
            (json.dumps({"data": []}).encode(), "expected one image"),
            (json.dumps({"data": [{"url": "http://x"}]}).encode(), "not a PNG"),
            (json.dumps({"data": [{"b64_json": "%%%"}]}).encode(), "not base64"),
            (
                json.dumps({"data": [{"b64_json": base64.b64encode(b"\xff\xd8\xff\xe0jpeg").decode()}]}).encode(),
                "not a PNG",
            ),
        ],
    )
    def test_an_answer_that_is_not_one_png_is_not_retried(self, engine_factory, body, match):
        engine = engine_factory(_Reply(body=body))

        with pytest.raises(NonRetryableError, match=match):
            asyncio.run(_client(engine.url).generate_image(_REQUEST))
        assert len(engine.requests) == 1


class TestConcurrency:
    def test_the_cap_bounds_requests_in_flight(self, engine_factory):
        engine = engine_factory(*[_Reply(delay_secs=0.2)] * 6)
        client = _client(engine.url, max_concurrency=2)

        async def run() -> list[bytes]:
            return await asyncio.gather(*(client.generate_image(_REQUEST) for _ in range(6)))

        assert asyncio.run(run()) == [_png(64, 32)] * 6
        assert engine.max_in_flight == 2


class TestFactory:
    def test_the_named_models_router_is_used(self, engine_factory):
        generator, policy = engine_factory(), engine_factory()
        args = SimpleNamespace(
            sglang_model_routers={"policy": _host_port(policy.url), "generator": _host_port(generator.url)},
            sglang_router_ip="127.0.0.1",
            sglang_router_port=_host_port(policy.url)[1],
        )

        client = image_generation_client(args, "generator", timeout_secs=5.0)
        asyncio.run(client.generate_image(_REQUEST))

        assert len(generator.requests) == 1 and policy.requests == []

    def test_an_unknown_model_never_falls_back_to_the_policy(self):
        args = SimpleNamespace(
            sglang_model_routers={"policy": ("127.0.0.1", 1)}, sglang_router_ip="127.0.0.1", sglang_router_port=1
        )

        with pytest.raises(NonRetryableError, match=r"'generator'.*\['policy'\]"):
            image_generation_client(args, "generator", timeout_secs=5.0)

    def test_an_ipv6_router_is_bracketed(self):
        args = SimpleNamespace(sglang_model_routers={"generator": ("::1", 30000)})

        client = image_generation_client(args, "generator", timeout_secs=5.0)

        assert client.url == "http://[::1]:30000/v1/images/generations"


def test_the_module_does_not_import_torch():
    code = (
        "import sys, miles.rollout.generate_utils.image_generation\n" "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def _host_port(url: str) -> tuple[str, int]:
    host, port = url.removeprefix("http://").rsplit(":", 1)
    return host, int(port)
