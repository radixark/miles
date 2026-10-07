"""Standalone session-server process: HTTP chassis + upstream proxy transport.

- ``SessionServer`` is a FastAPI app plus one shared aiohttp client; ``do_proxy`` forwards a request to the inference router (sglang or miles) — which does the load balancing to worker engines — and returns the raw result, or a 502 JSON error on transport failure.
- Session/TITO logic lives in ``core.SessionCore``; ``setup_session_routes`` (``sessions.py``) wires the HTTP routes to it.
- Standalone (own process, own event loop) so sessions also work with the SGLang Rust Router or any other backend, decoupled from the Miles Router.
- ``run_session_server`` is the subprocess entry point: fresh interpreter, so it configures logging and the process title itself, then serves uvicorn.
"""

import asyncio
import gc
import json
import logging
from contextlib import asynccontextmanager

import aiohttp
import setproctitle
import uvicorn
from fastapi import FastAPI

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.sessions import setup_session_routes
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)

# Request headers that must not be forwarded verbatim to the upstream backend.
_DROP_REQUEST_HEADERS = ("content-length", "transfer-encoding", "host")


class _TimedBytesPayload(aiohttp.BytesPayload):
    def __init__(self, body: bytes, *, timeout: float):
        super().__init__(body)
        self._write_timeout = timeout

    async def write(self, writer) -> None:
        await self.write_with_length(writer, None)

    async def write_with_length(self, writer, content_length: int | None) -> None:
        body = self._value if content_length is None else self._value[:content_length]
        # aiohttp's read deadline starts after upload; bound a stalled upload separately.
        await asyncio.wait_for(writer.write(body), timeout=self._write_timeout)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, config: SessionServerConfig):
        self.backend_url = config.backend_url
        self.app = FastAPI(lifespan=self._lifespan)

        # Every turn's backend reply is megabytes (per-token logprobs, R3). aiohttp parses it in C;
        # httpx's pure-Python client path took 35-40% of a saturated server's CPU.
        # Connecting (pool wait included) and each socket read; the payload bounds writing.
        self.timeout = aiohttp.ClientTimeout(total=None, connect=config.timeout, sock_read=config.timeout)
        self.client: aiohttp.ClientSession

        # `retract` may recompute earlier rows and must return full R3; all other
        # pause modes preserve prior rows and can request only the appended R3.
        self.use_addition_r3 = config.pause_generation_mode != "retract"
        setup_session_routes(self.app, self, config, use_addition_r3=self.use_addition_r3)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Bind the client to uvicorn's running loop before accepting requests.
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=1024, keepalive_timeout=5),
            timeout=self.timeout,
            trust_env=True,
        ) as client:
            self.client = client
            yield

    async def do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"

        headers = {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}

        try:
            # The body is read inside the try: a reply cut off mid-body is a transport error (502), as with httpx.
            # skip_auto_headers: a request without Content-Type reaches the backend without one (SGLang then
            # parses JSON), instead of aiohttp's application/octet-stream.
            async with self.client.request(
                request.method,
                url,
                data=_TimedBytesPayload(body, timeout=self.timeout.sock_read),
                headers=headers,
                allow_redirects=False,
                skip_auto_headers=("Content-Type",),
            ) as response:
                content = await response.read()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
            error_body = json.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"}).encode()
            return {
                "request_body": body,
                "response_body": error_body,
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status,
            # Lower-case keys, as httpx returned them: callers look up "content-type".
            "headers": {k.lower(): v for k, v in response.headers.items()},
        }


def run_session_server(config: SessionServerConfig):
    """Entry point to start the standalone session server as a subprocess."""
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("session_server")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    server = SessionServer(config)
    # Every record keeps its turn's full prompt `input_ids`, so a full collection walks O(turns^2) list items
    # per session (0.3-0.8 s pauses at 64 sessions) while reclaiming almost nothing; young collections still
    # free per-request cycles. Consider a full collection after 100 generation-1 collections instead of 10.
    gen0, gen1, _ = gc.get_threshold()
    gc.set_threshold(gen0, gen1, 100)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        config.host,
        config.port,
        config.backend_url,
    )
    uvicorn.run(server.app, host=config.host, port=config.port, log_level="info", access_log=False)


def main(argv: list[str] | None = None) -> None:
    run_session_server(parse_config_argv(SessionServerConfig, argv))


if __name__ == "__main__":
    main()
