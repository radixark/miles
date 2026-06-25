"""Standalone Session Server that proxies through the inference router.

This decouples session/TITO logic from the Miles Router, allowing sessions
to work with the SGLang Rust Router or any other backend.  Inference
requests are proxied through the router (sglang or miles), which handles
load balancing and forwarding to worker engines.
"""

import json
import logging

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI
from starlette.responses import Response

from miles.rollout.session.session_core import ProxyRequest, _proxy_result_to_core_response
from miles.rollout.session.sessions import setup_session_routes

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles).

    The request-handling logic lives in the transport-neutral ``SessionCore``
    (see ``session_core``); this class owns the FastAPI app, the httpx client,
    and the upstream proxy (``do_proxy``)."""

    def __init__(self, args, backend_url: str):
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", 600.0)
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024),
            timeout=httpx.Timeout(timeout),
        )

        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self.client.aclose)

        setup_session_routes(self.app, self, args)

    async def do_proxy(
        self,
        request: ProxyRequest,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        # ``request`` is a transport-neutral ProxyRequest (method + raw query),
        # not a fastapi.Request: the core drives the proxy from primitives. The
        # signature stays (request, path, body=None, headers=None) so tests can
        # still patch SessionServer.do_proxy and forward through it.
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"

        headers = {
            k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding", "host")
        }

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
            error_body = json.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"}).encode()
            return {
                "request_body": body,
                "response_body": error_body,
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

    def build_proxy_response(self, result: dict) -> Response:
        # Thin Starlette adapter over the core's passthrough builder; kept on
        # SessionServer because the unit tests call it directly.
        core_response = _proxy_result_to_core_response(result)
        return Response(
            content=core_response.body,
            status_code=core_response.status_code,
            headers=core_response.headers,
            media_type=core_response.media_type,
        )


def run_session_server(args, backend_url: str):
    """Entry point to start the standalone session server as a subprocess."""
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    server = SessionServer(args, backend_url)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    # Single uvicorn worker on purpose: extra workers would each own a separate
    # SessionRegistry + asyncio.Lock, so a session_id could land on a process that
    # doesn't own it. Multi-process needs sticky session ownership and is deferred.
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")
