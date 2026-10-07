"""Standalone session-server process: HTTP chassis + upstream proxy transport.

- ``SessionServer`` is a FastAPI app plus one shared aiohttp client; ``do_proxy`` forwards a request to the inference router (sglang or miles) — which does the load balancing to worker engines — and returns the raw result, or a 502 JSON error on transport failure.
- Session/TITO logic lives in ``core.SessionCore``; ``setup_session_routes`` (``sessions.py``) wires the HTTP routes to it.
- Standalone (own process, own event loop) so sessions also work with the SGLang Rust Router or any other backend, decoupled from the Miles Router.
- ``run_session_server`` is the subprocess entry point: fresh interpreter, so it configures logging and the process title itself, then serves uvicorn.
"""

import json
import logging

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


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, config: SessionServerConfig):
        self.backend_url = config.backend_url
        self.app = FastAPI()

        # Every turn's backend reply is megabytes (per-token logprobs, R3). aiohttp parses it in C;
        # httpx's pure-Python client path took 35-40% of a saturated server's CPU.
        # Bounds as before: connecting (pool wait included) and each socket read.
        self.timeout = aiohttp.ClientTimeout(total=None, connect=config.timeout, sock_read=config.timeout)
        # A ClientSession binds to the running loop, so the first proxy call opens it.
        self.client: aiohttp.ClientSession | None = None

        # Close the connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self._close_client)

        # `retract` may recompute earlier rows and must return full R3; all other
        # pause modes preserve prior rows and can request only the appended R3.
        self.use_addition_r3 = config.pause_generation_mode != "retract"
        setup_session_routes(self.app, self, config, use_addition_r3=self.use_addition_r3)

    def _get_client(self) -> aiohttp.ClientSession:
        if self.client is None:
            # httpx defaults kept: 5 s keep-alive, no redirects followed, proxy settings from the environment.
            self.client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=1024, keepalive_timeout=5),
                timeout=self.timeout,
                trust_env=True,
            )
        return self.client

    async def _close_client(self) -> None:
        if self.client is not None:
            await self.client.close()

    async def do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"

        headers = {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}

        try:
            # The body is read inside the try: a reply cut off mid-body is a transport error (502), as with httpx.
            # skip_auto_headers: a request without Content-Type reaches the backend without one (SGLang then
            # parses JSON), instead of aiohttp's application/octet-stream.
            async with self._get_client().request(
                request.method,
                url,
                data=body,
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
