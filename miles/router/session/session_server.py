"""Standalone Session Server that proxies to any backend router (sglang or miles).

This decouples session/TITO logic from the Miles Router, allowing sessions
to work with the SGLang Rust Router or any other backend that exposes
standard ``/v1/chat/completions`` endpoints.
"""

import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.router.session.sessions import setup_session_routes

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests to a backend router (SGLang Rust Router or Miles Router)."""

    def __init__(self, args, backend_url: str):
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", None)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
        )

        setup_session_routes(self.app, self, args)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        url = f"{self.backend_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        response = await self.client.request(request.method, url, content=body, headers=headers)
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
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)


def run_session_server(args):
    """Entry point to start the standalone session server as a subprocess."""
    backend_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    server = SessionServer(args, backend_url)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")
