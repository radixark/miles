# doc-dev: docs/developer/multi-process-session-server.md
"""Headless session worker process — owns one shard of the multi-process data plane.

- Owns one shard's session state: its own ``SessionCore`` (registry + tokenizer) plus its own httpx ``ProxyBackend``.
- Speaks IPC, not HTTP: ``SessionWorker.handle`` decodes a request envelope, drives the matching ``SessionCore`` op, and encodes the returned Starlette ``Response`` (status + headers + body bytes) back.
- A ``SessionError`` from the core becomes the same ``error_response`` the single-process FastAPI handler produces, so the workers=N and workers=1 error paths stay byte-identical.
- ``ProxyBackend.do_proxy`` filters the same request headers as ``SessionServer.do_proxy`` (exercised by the equivalence tests) and turns a transport failure into a 502 result.
- ``run_worker`` is the ``multiprocessing.Process`` target: names the process, arms PR_SET_PDEATHSIG, then serves the socket until the channel closes.

The worker trusts the router's stable-hash routing and does not re-derive ownership.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import signal
import sys

import httpx
import setproctitle
from starlette.responses import Response

from miles.rollout.session.core import ProxyRequest, build_session_core, error_response
from miles.rollout.session.errors import SessionError
from miles.rollout.session.ipc import (
    OP_CHAT,
    OP_CREATE,
    OP_DELETE,
    OP_GET,
    OP_HEALTH,
    OP_PROXY,
    decode_envelope,
    encode_envelope,
    open_unix_channel,
)

logger = logging.getLogger(__name__)

# Request headers not forwarded verbatim upstream (mirrors SessionServer.do_proxy;
# the workers=N and workers=1 paths must filter identically — pinned by the
# multi-process equivalence tests).
_DROP_REQUEST_HEADERS = ("content-length", "transfer-encoding", "host")


class ProxyBackend:
    """Minimal httpx-only proxy backend for a worker (no FastAPI app)."""

    def __init__(self, backend_url: str, *, timeout: float = 600.0, max_connections: int = 1024):
        self.backend_url = backend_url
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

    async def do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"
        headers = {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}
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

    async def aclose(self) -> None:
        await self.client.aclose()


class SessionWorker:
    """Decodes IPC request envelopes, drives a SessionCore, encodes replies."""

    def __init__(self, core):
        self.core = core

    async def handle(self, payload: bytes) -> bytes:
        meta, body = decode_envelope(payload)
        try:
            response = await self._dispatch(meta, body)
        except SessionError as exc:
            response = error_response(exc)
        return encode_envelope(
            {"status": response.status_code, "headers": dict(response.headers)}, response.body or b""
        )

    async def _dispatch(self, meta: dict, body: bytes) -> Response:
        op = meta["op"]
        if op == OP_HEALTH:
            return await self.core.health()
        if op == OP_CREATE:
            return await self.core.create_session(meta["session_id"])
        if op == OP_GET:
            return await self.core.get_session(meta["session_id"])
        if op == OP_DELETE:
            return await self.core.delete_session(meta["session_id"])
        if op == OP_CHAT:
            return await self.core.chat_completions(
                meta["session_id"], method=meta["method"], query=meta["query"], headers=meta["headers"], body=body
            )
        if op == OP_PROXY:
            return await self.core.proxy(
                meta["session_id"],
                meta["path"],
                method=meta["method"],
                query=meta["query"],
                headers=meta["headers"],
                body=body,
            )
        raise ValueError(f"unknown op: {op!r}")


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL this worker if the parent (supervisor) dies (Linux)."""
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        logger.warning("Failed to set PR_SET_PDEATHSIG; worker may outlive a crashed parent")


async def _serve(args, backend_url: str, sock) -> None:
    backend = ProxyBackend(backend_url, timeout=getattr(args, "miles_router_timeout", 600.0))
    worker = SessionWorker(build_session_core(backend, args))
    closed = asyncio.Event()
    await open_unix_channel(sock, request_handler=worker.handle, on_close=closed.set)
    try:
        await closed.wait()
    finally:
        await backend.aclose()


def run_worker(args, backend_url: str, sock, worker_index: int) -> None:
    """``multiprocessing.Process`` target: serve one session shard over ``sock``."""
    setproctitle.setproctitle(f"miles-session-worker-{worker_index}")
    _set_pdeathsig()
    asyncio.run(_serve(args, backend_url, sock))
