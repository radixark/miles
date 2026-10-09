"""Recorded-session routes: bind, export, delete (tenant key) and the OpenAI chat route (session id)."""

import json

from fastapi import FastAPI, Request

from miles.tinker.core.tinker_session_server import TrajectoryCollector
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


def setup_session_routes(app: FastAPI, collector: TrajectoryCollector, max_body_bytes: int) -> None:
    """Mount bind, chat, export and delete; build_app's handler maps their errors."""

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

    @app.post("/oai/sessions/{session_id}/v1/chat/completions")
    async def session_chat(session_id: str, request: Request):
        """One recorded turn of a bound session; the unguessable session id is the credential."""
        body = await _json_body(request, max_body_bytes)
        return chat_completion_json(body, await collector.complete(session_id, parse_chat_request(body)))

    @app.get("/oai/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        """Export {session_id, model_path, max_trim_tokens, turns: [ids, logprobs, finish_reason, ...]}; owner only."""
        return collector.get_session(session_id, _tenant(request))

    @app.delete("/oai/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        """Free the session and its turns, cancelling a sample still running; bearer must match the owner."""
        collector.delete_session(session_id, _tenant(request))
        return {"session_id": session_id, "deleted": True}
