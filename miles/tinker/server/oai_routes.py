"""OpenAI-compatible routes served by the token trajectory collector, mounted next to the Tinker API.

Skeleton: route bodies land with the collector.

Reused, not reimplemented: ``build_app`` (all Tinker routes and the UserInputError→400 / OwnershipError→403
handlers) and ``_tenant`` from ``miles.tinker.server.app``. Only the routes below and a 404 handler are new.
"""

from fastapi import FastAPI, Request

from miles.tinker.core.service import TinkerService
from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.server.app import build_app


def _optional_tenant(request: Request) -> str | None:
    """_tenant, but None instead of UserInputError when no key is present (a pre-bound session serves the harness's dummy key)."""
    raise NotImplementedError


def _install_oai_routes(app: FastAPI, collector: TrajectoryCollector) -> None:
    """Mount the OpenAI-compatible routes: Tinker-shaped stateless ones under /oai/api/v1, recorded ones under /oai/sessions/{sid}/v1; adds the UnknownSessionError→404 handler."""

    @app.post("/oai/api/v1/chat/completions")
    async def oai_chat_completions(request: Request):
        """Stateless chat completion in Tinker's OpenAI-compatible shape; not recorded; bearer required (_tenant)."""
        raise NotImplementedError

    @app.post("/oai/api/v1/completions")
    async def oai_completions(request: Request):
        """Stateless raw-prompt completion in Tinker's OpenAI-compatible shape; not recorded; bearer required."""
        raise NotImplementedError

    @app.post("/oai/sessions/{session_id}")
    async def oai_bind_session(session_id: str, request: Request):
        """Pre-bind a session to {model} or {sampling_session_id}; bearer required."""
        raise NotImplementedError

    @app.post("/oai/sessions/{session_id}/v1/chat/completions")
    async def oai_session_chat_completions(session_id: str, request: Request):
        """Recorded chat completion; a new session_id with a valid bearer auto-registers, a pre-bound one accepts the harness's dummy key (_optional_tenant)."""
        raise NotImplementedError

    @app.post("/oai/sessions/{session_id}/v1/completions")
    async def oai_session_completions(session_id: str, request: Request):
        """Recorded raw-prompt completion on the same session."""
        raise NotImplementedError

    @app.get("/oai/sessions/{session_id}")
    async def oai_get_session(session_id: str, request: Request):
        """Export the session's turns (input_ids, output_ids, logprobs, finish_reason); bearer must match the owner."""
        raise NotImplementedError

    @app.delete("/oai/sessions/{session_id}")
    async def oai_delete_session(session_id: str, request: Request):
        """Free the session and its turns; bearer must match the owner."""
        raise NotImplementedError


def build_app_with_collector(service: TinkerService, collector: TrajectoryCollector) -> FastAPI:
    """build_app plus the /oai routes served by the token trajectory collector; the Tinker routes are untouched."""
    app = build_app(service)
    _install_oai_routes(app, collector)
    return app
