"""OpenAI-compatible routes served by the token trajectory collector, mounted next to the Tinker API.

Skeleton: route bodies land with the collector. ``build_app`` itself is untouched; ``build_app_with_collector``
wraps it and adds the ``/oai`` routes (Tinker-shaped stateless ones under ``/oai/api/v1``, recorded sessions
under ``/oai/sessions/{sid}/v1``).
"""

from fastapi import FastAPI, Request

from miles.tinker.core.service import TinkerService
from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.server.app import build_app


def _optional_tenant(request: Request) -> str | None:
    """Bearer or X-API-Key when present; None lets a pre-bound session serve a harness that only has a dummy key."""
    raise NotImplementedError


def _install_oai_routes(app: FastAPI, collector: TrajectoryCollector) -> None:
    """Mount the OpenAI-compatible routes: Tinker-shaped stateless ones under /oai/api/v1, recorded ones under /oai/sessions/{sid}/v1."""

    @app.post("/oai/api/v1/chat/completions")
    async def oai_chat_completions(request: Request):
        """Stateless chat completion in Tinker's OpenAI-compatible shape; not recorded."""
        raise NotImplementedError

    @app.post("/oai/api/v1/completions")
    async def oai_completions(request: Request):
        """Stateless raw-prompt completion in Tinker's OpenAI-compatible shape; not recorded."""
        raise NotImplementedError

    @app.post("/oai/sessions/{session_id}")
    async def oai_bind_session(session_id: str, request: Request):
        """Pre-bind a session to {model}; bearer required."""
        raise NotImplementedError

    @app.post("/oai/sessions/{session_id}/v1/chat/completions")
    async def oai_session_chat_completions(session_id: str, request: Request):
        """Recorded chat completion; a new session_id with a valid bearer auto-registers, a pre-bound one accepts the harness's dummy key."""
        raise NotImplementedError

    @app.post("/oai/sessions/{session_id}/v1/completions")
    async def oai_session_completions(session_id: str, request: Request):
        """Recorded raw-prompt completion on the same session."""
        raise NotImplementedError

    @app.get("/oai/sessions/{session_id}")
    async def oai_get_session(session_id: str, request: Request):
        """Export the session's turns (input_ids, output_ids, logprobs, finish_reason, prefix_ok); bearer must match the owner."""
        raise NotImplementedError

    @app.delete("/oai/sessions/{session_id}")
    async def oai_delete_session(session_id: str, request: Request):
        """Free the session and its turns."""
        raise NotImplementedError


def build_app_with_collector(service: TinkerService, collector: TrajectoryCollector) -> FastAPI:
    """build_app plus the /oai routes served by the token trajectory collector; the Tinker routes are untouched."""
    app = build_app(service)
    _install_oai_routes(app, collector)
    return app
