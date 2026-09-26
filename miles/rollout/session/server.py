"""Standalone session-server process: HTTP chassis + upstream proxy transport.

- ``SessionServer`` is a FastAPI app plus one shared httpx client; ``do_proxy`` forwards a request to the inference router (sglang or miles) — which does the load balancing to worker engines — and returns the raw result, or a 502 JSON error on transport failure.
- Session/TITO logic lives in ``core.SessionCore``; ``setup_session_routes`` (``sessions.py``) wires the HTTP routes to it.
- Standalone (own process, own event loop) so sessions also work with the SGLang Rust Router or any other backend, decoupled from the Miles Router.
- ``run_session_server`` is the subprocess entry point: fresh interpreter, so it configures logging and the process title itself, then serves uvicorn.
"""

import asyncio
import json
import logging

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.request_policy import RolloutRequestContext, prepare_rollout_request
from miles.rollout.session.sessions import setup_session_routes
from miles.utils.function_registry import load_function
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)

# Request headers that must not be forwarded verbatim to the upstream backend.
_DROP_REQUEST_HEADERS = ("content-length", "transfer-encoding", "host")


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, config: SessionServerConfig):
        self.config = config
        self.backend_url = config.backend_url
        self.request_hook = load_function(config.custom_rollout_request_hook_path)
        self.app = FastAPI()

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024),
            timeout=httpx.Timeout(config.timeout),
        )

        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self.client.aclose)

        # `retract` may recompute earlier rows and must return full R3; all other
        # pause modes preserve prior rows and can request only the appended R3.
        self.use_addition_r3 = config.pause_generation_mode != "retract"
        setup_session_routes(self.app, self, config, use_addition_r3=self.use_addition_r3)

    async def do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        deadline = asyncio.timeout(self.config.timeout)
        try:
            async with deadline:
                return await self._do_proxy(request, path, body=body, headers=headers)
        except TimeoutError:
            if not deadline.expired():
                raise
            logger.warning("Proxy request deadline exceeded for %s %s", request.method, path)
            return _proxy_error(body, "backend transport error: request deadline exceeded")

    async def _do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"

        headers = {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}
        max_attempts = 1
        retry_interval = 1.0
        if request.session_id is not None and self.request_hook is not None:
            prepared = await prepare_rollout_request(
                self.request_hook,
                self.config.custom_rollout_request_hook_args,
                RolloutRequestContext(session_id=request.session_id),
                payload=json.loads(body),
                headers=headers,
            )
            headers = prepared["headers"]
            body = json.dumps(prepared["payload"], ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
            max_attempts = prepared["max_attempts"]
            retry_interval = prepared["retry_interval"]

        response = None
        for attempt in range(max_attempts):
            try:
                response = await self.client.request(request.method, url, content=body, headers=headers)
            except httpx.TransportError as exc:
                if not _transport_error_is_safe_to_retry(exc) or attempt + 1 == max_attempts:
                    logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
                    return _proxy_error(body, f"backend transport error: {type(exc).__name__}: {exc}")
            else:
                if response.status_code not in (409, 429) or attempt + 1 == max_attempts:
                    break
                await response.aread()
            await asyncio.sleep(retry_interval)

        assert response is not None
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }


def _transport_error_is_safe_to_retry(exc: httpx.TransportError) -> bool:
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))


def _proxy_error(request_body: bytes, message: str) -> dict:
    return {
        "request_body": request_body,
        "response_body": json.dumps({"error": message}).encode(),
        "status_code": 502,
        "headers": {"content-type": "application/json"},
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
