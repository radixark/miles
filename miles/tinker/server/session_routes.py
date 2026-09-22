"""Recorded-session routes: bind, export, delete (tenant key) plus one chat route per API adapter (OpenAI today)."""

import json
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.tinker.core.tinker_session_server import (
    SessionError,
    SessionNotFoundError,
    TrajectoryCollector,
    TurnRequest,
    TurnResult,
)
from miles.tinker.core.types import UserInputError
from miles.tinker.server.app import _tenant
from miles.tinker.server.oai_shapes import chat_completion_json, parse_chat_request

# default for --tinker-session-max-body-bytes; bodies are parsed synchronously on the shared loop
MAX_BODY_BYTES = 16 * 1024 * 1024


async def _json_body(request: Request, max_body_bytes: int = MAX_BODY_BYTES) -> dict:
    """The JSON object body as a dict; bad JSON is a UserInputError (400); over-cap bodies are refused unbuffered."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_body_bytes:
        raise UserInputError(f"request body of {declared} bytes exceeds the {max_body_bytes}-byte limit")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():  # stop reading at the cap instead of buffering the whole body first
        size += len(chunk)
        if size > max_body_bytes:
            raise UserInputError(f"request body exceeds the {max_body_bytes}-byte limit")
        chunks.append(chunk)
    body = b"".join(chunks)
    if not body:
        return {}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise UserInputError(f"invalid JSON body: {error}") from error
    if not isinstance(payload, dict):
        raise UserInputError("request body must be a JSON object")
    return payload


# chat dialects: path suffix -> (body -> TurnRequest, (body, TurnResult) -> JSON); Anthropic: /v1/messages
CHAT_ADAPTERS: dict[str, tuple[Callable[[dict], TurnRequest], Callable[[dict, TurnResult], dict]]] = {
    "/v1/chat/completions": (parse_chat_request, chat_completion_json),
}


def _mount_chat_route(app: FastAPI, collector: TrajectoryCollector, suffix: str, parse, render, max_body_bytes: int):
    """POST /oai/sessions/{sid}{suffix}: one recorded Turn of a bound session; the session id is the credential."""

    @app.post(f"/oai/sessions/{{session_id}}{suffix}")
    async def session_chat(session_id: str, request: Request):
        body = await _json_body(request, max_body_bytes)
        return render(body, await collector.complete(session_id, parse(body)))


def setup_session_routes(app: FastAPI, collector: TrajectoryCollector, max_body_bytes: int = MAX_BODY_BYTES) -> None:
    """Mount bind/export/delete, one chat route per CHAT_ADAPTERS entry, and the SessionError handler (its status)."""

    @app.exception_handler(SessionError)
    async def _session_error(request: Request, error: SessionError):
        hint = ""
        if isinstance(error, SessionNotFoundError):
            session_id = request.path_params.get("session_id", "{sid}")
            hint = f"; bind it first with POST /oai/sessions/{session_id} (tenant key + sampling_session_id)"
        return JSONResponse(status_code=error.status_code, content={"error": f"{error}{hint}"})

    @app.post("/oai/sessions/{session_id}")
    async def create_session(session_id: str, request: Request):
        """Bind the session to a Tinker sampling session (optional max_datum_tokens); the tenant's key is required."""
        tenant = _tenant(request)
        payload = await _json_body(request, max_body_bytes)
        session = collector.create_session(
            session_id,
            tenant,
            sampling_session_id=payload.get("sampling_session_id"),
            max_datum_tokens=payload.get("max_datum_tokens"),
        )
        return {"session_id": session.session_id, "model_path": session.model_path}

    for suffix, (parse, render) in CHAT_ADAPTERS.items():
        _mount_chat_route(app, collector, suffix, parse, render, max_body_bytes)

    @app.get("/oai/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        """Export {session_id, model_path, max_trim_tokens, turns: [ids, logprobs, finish_reason, ...]}; owner only."""
        return collector.get_session(session_id, _tenant(request))

    @app.delete("/oai/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        """Free the session and its turns, cancelling a sample still running; bearer must match the owner."""
        collector.delete_session(session_id, _tenant(request))
        return {"session_id": session_id, "deleted": True}
