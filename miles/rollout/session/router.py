# doc-dev: docs/developer/multi-process-session-server.md
"""Thin client-facing router for the multi-process session server.

- The sole HTTP listener: a FastAPI app over N ``IpcChannel``s, one per worker.
- Routes every op of a session through ``SessionRouter.channel_for`` — a stable hash of ``session_id`` (``worker_index_for_session``) — so create/get/chat/delete land on the same worker.
- Relays the worker's response bytes back verbatim; it never parses a body.
- ``POST /sessions`` mints the id first (uuid4 hex) so it can route by it, then dispatches the create to the owning worker.
- Error mapping in ``dispatch``: channel down → 503, worker handler raised → 502, malformed reply envelope → uncaught 500 (a worker protocol violation).
- ``GET /health`` pings every worker and reports 503 unless all reply healthy within the timeout.

Imports neither ``core`` nor ``worker``, so the router process never loads the tokenizer/transformers stack — it only moves bytes.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import FastAPI, Request
from starlette.responses import Response

from miles.rollout.session.ipc import (
    OP_CHAT,
    OP_CREATE,
    OP_DELETE,
    OP_GET,
    OP_HEALTH,
    OP_PROXY,
    IpcChannelClosed,
    IpcError,
    decode_envelope,
    encode_request,
)
from miles.rollout.session.sharding import worker_index_for_session

_JSON = "application/json"
_HEALTH_TIMEOUT = 5.0


def _err(status_code: int, message: str) -> Response:
    return Response(content=json.dumps({"error": message}).encode(), status_code=status_code, media_type=_JSON)


def _reply_to_response(reply: bytes) -> Response:
    """Relay a worker REPLY envelope as an HTTP Response verbatim.

    ``meta`` must carry ``status`` + ``headers`` (worker contract); ``body`` is
    passed through unparsed, and a missing key raises KeyError (= worker protocol
    violation, deliberately a 500). ``headers`` already carries content-type, so
    no ``media_type`` is passed — that would duplicate content-length/-type.
    """
    meta, body = decode_envelope(reply)
    return Response(content=body, status_code=meta["status"], headers=meta["headers"])


class SessionRouter:
    """Routes requests to per-worker IPC channels by stable hash of session_id."""

    def __init__(self, channels: list, session_server_instance_id=None):
        if not channels:
            raise ValueError("SessionRouter requires at least one worker channel")
        self.channels = channels
        self.n_worker = len(channels)
        self.instance_id = session_server_instance_id

    def channel_for(self, session_id: str):
        """The sole routing point: the worker channel owning ``session_id``.

        Every op for a session must resolve here so create/get/chat/delete land
        on one worker; divergent hashing would silently mis-route.
        """
        return self.channels[worker_index_for_session(session_id, self.n_worker)]

    async def dispatch(self, channel, payload: bytes) -> Response:
        """Send one request to ``channel`` and map the outcome to a Response.

        Channel down (IpcChannelClosed) → 503; worker handler raised (IpcError)
        → 502; a malformed reply envelope is not caught → 500. No per-request
        timeout: covers worker death, not a live-but-hung worker.
        """
        try:
            reply = await channel.request(payload)
        except IpcChannelClosed:
            return _err(503, "session worker unavailable")
        except IpcError as exc:
            return _err(502, f"session worker error: {exc}")
        return _reply_to_response(reply)

    async def healthy(self) -> bool:
        async def ping(channel) -> bool:
            meta, _ = decode_envelope(await channel.request(encode_request(OP_HEALTH)))
            return meta["status"] == 200

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(ping(ch) for ch in self.channels)), timeout=_HEALTH_TIMEOUT
            )
        except (asyncio.TimeoutError, IpcChannelClosed, IpcError):
            return False
        return all(results)


def build_router_app(channels: list, session_server_instance_id=None) -> FastAPI:
    router = SessionRouter(channels, session_server_instance_id)
    app = FastAPI()

    @app.get("/health")
    async def health():
        if not await router.healthy():
            return _err(503, "one or more session workers unhealthy")
        body = {"status": "ok"}
        if router.instance_id is not None:
            body["session_server_instance_id"] = router.instance_id
        return Response(content=json.dumps(body).encode(), status_code=200, media_type=_JSON)

    @app.post("/sessions")
    async def create_session():
        session_id = uuid.uuid4().hex
        return await router.dispatch(router.channel_for(session_id), encode_request(OP_CREATE, session_id=session_id))

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return await router.dispatch(router.channel_for(session_id), encode_request(OP_GET, session_id=session_id))

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        return await router.dispatch(router.channel_for(session_id), encode_request(OP_DELETE, session_id=session_id))

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        body = await request.body()
        payload = encode_request(
            OP_CHAT,
            session_id=session_id,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )
        return await router.dispatch(router.channel_for(session_id), payload)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        body = await request.body()
        payload = encode_request(
            OP_PROXY,
            session_id=session_id,
            path=path,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )
        return await router.dispatch(router.channel_for(session_id), payload)

    return app
