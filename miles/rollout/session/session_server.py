"""Standalone Session Server that proxies to SGLang worker engines directly.

This decouples session/TITO logic from the Miles Router, allowing sessions
to work with the SGLang Rust Router or any other backend.  Requests are
proxied directly to SGLang worker engines (not the Rust Router) so that
the full response including ``meta_info`` is preserved.
"""

import itertools
import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.sessions import setup_session_routes

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests directly to SGLang worker engines."""

    def __init__(self, args, worker_urls: list[str]):
        self.worker_urls = worker_urls
        self._worker_cycle = itertools.cycle(worker_urls)
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
        worker_url = next(self._worker_cycle)
        url = f"{worker_url}/{path}"

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


def run_session_server(args, worker_urls: list[str]):
    """Entry point to start the standalone session server as a subprocess."""
    server = SessionServer(args, worker_urls)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        worker_urls,
    )
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")
