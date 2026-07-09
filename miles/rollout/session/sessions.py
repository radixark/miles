"""Single-process FastAPI adapter for the session server.

Thin layer: converts each HTTP request to primitive inputs, calls
``SessionCore``. All session/TITO logic lives in ``core``.
"""

import logging

from fastapi import Request

from miles.rollout.session.core import build_session_core, error_response
from miles.rollout.session.errors import SessionError

logger = logging.getLogger(__name__)


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    core = build_session_core(backend, args)

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return error_response(exc)

    @app.get("/health")
    async def health():
        return await core.health()

    @app.post("/sessions")
    async def create_session():
        return await core.create_session()

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return await core.get_session(session_id)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        return await core.delete_session(session_id)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        body = await request.body()
        return await core.chat_completions(
            session_id,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        body = await request.body()
        return await core.proxy(
            session_id,
            path,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )
