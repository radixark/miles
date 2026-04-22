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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.sessions import setup_session_routes

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

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
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
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
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)


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
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")
