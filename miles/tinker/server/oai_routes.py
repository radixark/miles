"""The four recorded-session routes (bind, chat, turns, delete) beside the Tinker API, with 404/429/502 handlers."""

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.tinker.core.tinker_session_server import (
    SamplingBackendError,
    SessionLimitError,
    TrajectoryCollector,
    UnknownSessionError,
)
from miles.tinker.core.types import UserInputError
from miles.tinker.server.app import _tenant

# default for --tinker-session-max-body-bytes; bodies are parsed synchronously on the shared loop
MAX_BODY_BYTES = 16 * 1024 * 1024


def _optional_tenant(request: Request) -> str | None:
    """_tenant, but None when no key is present (a pre-bound session serves the harness's dummy key)."""
    try:
        return _tenant(request)
    except UserInputError:
        return None


async def _json_body(request: Request, max_body_bytes: int = MAX_BODY_BYTES) -> dict:
    """The JSON object body as a dict; bad JSON is a UserInputError (400); bodies over max_body_bytes are refused."""
    body = await request.body()
    if len(body) > max_body_bytes:
        raise UserInputError(f"request body of {len(body)} bytes exceeds the {max_body_bytes}-byte limit")
    if not body:
        return {}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise UserInputError(f"invalid JSON body: {error}") from error
    if not isinstance(payload, dict):
        raise UserInputError("request body must be a JSON object")
    return payload


def install_session_routes(app: FastAPI, collector: TrajectoryCollector, max_body_bytes: int = MAX_BODY_BYTES) -> None:
    """Mount the four /oai/sessions routes (bodies capped at max_body_bytes) and their 404 / 429 / 502 handlers."""

    @app.exception_handler(UnknownSessionError)
    async def _unknown_session(request: Request, error: UnknownSessionError):
        return JSONResponse(status_code=404, content={"error": str(error)})

    @app.exception_handler(SessionLimitError)
    async def _session_limit(request: Request, error: SessionLimitError):
        return JSONResponse(status_code=429, content={"error": str(error)})

    @app.exception_handler(SamplingBackendError)
    async def _backend_error(request: Request, error: SamplingBackendError):
        return JSONResponse(status_code=502, content={"error": str(error)})

    @app.post("/oai/sessions/{session_id}")
    async def bind_session(session_id: str, request: Request):
        """Pin the session to a tinker:// path or sampling_session_id (optional max_datum_tokens); bearer required."""
        tenant = _tenant(request)
        payload = await _json_body(request, max_body_bytes)
        session = collector.bind(
            session_id,
            tenant,
            model=payload.get("model"),
            sampling_session_id=payload.get("sampling_session_id"),
            max_datum_tokens=payload.get("max_datum_tokens"),
        )
        return {"session_id": session.session_id, "model_path": session.model_path}

    @app.post("/oai/sessions/{session_id}/v1/chat/completions")
    async def session_chat_completions(session_id: str, request: Request):
        """Chat completion recorded as one Turn; dummy key for pre-bound sessions, a real bearer auto-registers."""
        tenant = _optional_tenant(request)
        return await collector.chat(await _json_body(request, max_body_bytes), session_id=session_id, tenant=tenant)

    @app.get("/oai/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        """Export {session_id, model_path, turns: [ids, logprobs, finish_reason, inherits]}; owner only."""
        return collector.trajectory(session_id, _tenant(request))

    @app.delete("/oai/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        """Free the session and its turns; bearer must match the owner."""
        collector.delete(session_id, _tenant(request))
        return {"session_id": session_id, "deleted": True}
