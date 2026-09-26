"""Recorded-session routes: bind, export, delete (tenant key) plus one chat route per API adapter (OpenAI today)."""

import json
from collections.abc import Callable

from fastapi import FastAPI, Request

from miles.tinker.core.tinker_session_server import TrajectoryCollector, TurnRequest, TurnResult
from miles.tinker.core.types import UserInputError
from miles.tinker.server.app import _tenant
from miles.tinker.server.oai_shapes import chat_completion_json, parse_chat_request


async def _json_body(request: Request, max_body_bytes: int) -> dict:
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
    except (ValueError, RecursionError) as error:  # bad JSON, bytes that are not UTF-8, or nesting too deep to parse
        raise UserInputError(f"invalid JSON body: {error}") from None
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


def setup_session_routes(app: FastAPI, collector: TrajectoryCollector, max_body_bytes: int) -> None:
    """Mount bind/export/delete and one chat route per CHAT_ADAPTERS entry; build_app's handler maps their errors."""

    @app.post("/oai/sessions/{session_id}")
    async def create_session(session_id: str, request: Request):
        """Bind to a Tinker sampling session (tenant key); answers the effective per-datum cap the client must keep."""
        tenant = _tenant(request)
        payload = await _json_body(request, max_body_bytes)
        session = collector.create_session(
            session_id,
            tenant,
            sampling_session_id=payload.get("sampling_session_id"),
            max_datum_tokens=payload.get("max_datum_tokens"),
        )
        return {
            "session_id": session.session_id,
            "model_path": session.model_path,
            "max_datum_tokens": collector.datum_budget(session),
        }

    for suffix, (parse, render) in CHAT_ADAPTERS.items():
        _mount_chat_route(app, collector, suffix, parse, render, max_body_bytes)

    @app.get("/oai/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        """Export {session_id, model_path, turns: [ids, logprobs, finish_reason, parent, ...]}; owner only."""
        return collector.get_session(session_id, _tenant(request))

    @app.delete("/oai/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        """Free the session and its turns, cancelling a sample still running; bearer must match the owner."""
        collector.delete_session(session_id, _tenant(request))
        return {"session_id": session_id, "deleted": True}
