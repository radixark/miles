"""The four recorded-session routes the Harbor × cookbook example needs, mounted next to the Tinker API.

Skeleton: route bodies land with the collector.

Dependency decision: the gateway stays Tinker-wire only plus these four routes. No OpenAI-parity surface
(stateless ``/oai/api/v1/*``, ``/completions``, streaming) in stage 1; the agent harness only needs a recorded
chat endpoint. Reused, not reimplemented: ``build_app`` (all Tinker routes and the UserInputError→400 /
OwnershipError→403 handlers) and ``_tenant`` from ``miles.tinker.server.app``.
"""

from fastapi import FastAPI, Request

from miles.tinker.core.service import TinkerService
from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.server.app import build_app


def _optional_tenant(request: Request) -> str | None:
    """_tenant, but None instead of UserInputError when no key is present (a pre-bound session serves the harness's dummy key)."""
    raise NotImplementedError


def _install_session_routes(app: FastAPI, collector: TrajectoryCollector) -> None:
    """Mount the four /oai/sessions routes and the UnknownSessionError→404 handler."""

    @app.post("/oai/sessions/{session_id}")
    async def bind_session(session_id: str, request: Request):
        """Pin the session to {model: tinker://M/sampler_weights/V} (or {sampling_session_id}); bearer required; the example's step 1."""
        raise NotImplementedError

    @app.post("/oai/sessions/{session_id}/v1/chat/completions")
    async def session_chat_completions(session_id: str, request: Request):
        """OpenAI chat completion recorded as one Turn; a pre-bound session accepts the harness's dummy key, a new id with a valid bearer auto-registers; the example's step 2."""
        raise NotImplementedError

    @app.get("/oai/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        """Export {session_id, model_path, turns: [{input_ids, output_ids, logprobs, finish_reason}]}; bearer must match the owner; the example's step 3."""
        raise NotImplementedError

    @app.delete("/oai/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        """Free the session and its turns; bearer must match the owner."""
        raise NotImplementedError


def build_app_with_collector(service: TinkerService, collector: TrajectoryCollector) -> FastAPI:
    """build_app plus the four session routes; the Tinker routes are untouched."""
    app = build_app(service)
    _install_session_routes(app, collector)
    return app
