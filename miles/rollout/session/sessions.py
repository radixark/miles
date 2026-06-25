import logging

from fastapi import Request
from starlette.responses import Response

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_core import (
    CoreResponse,
    SessionCore,
    _dump_request_body,
    _parse_and_validate_response,
    _parse_request_body,
)
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

# Re-exported so existing tests can import the pure helpers from this module.
__all__ = [
    "setup_session_routes",
    "_parse_request_body",
    "_dump_request_body",
    "_parse_and_validate_response",
]

logger = logging.getLogger(__name__)


def _to_starlette_response(core_response: CoreResponse) -> Response:
    return Response(
        content=core_response.body,
        status_code=core_response.status_code,
        headers=core_response.headers,
        media_type=core_response.media_type,
    )


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    session_server_instance_id = getattr(args, "session_server_instance_id", None)

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
        allowed_append_roles=getattr(args, "tito_allowed_append_roles", None),
    )

    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)
    core = SessionCore(backend, registry, args, session_server_instance_id)

    @app.get("/health")
    async def health():
        return _to_starlette_response(await core.health())

    @app.post("/sessions")
    async def create_session():
        return _to_starlette_response(await core.create_session())

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return _to_starlette_response(await core.get_session(session_id))

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        return _to_starlette_response(await core.delete_session(session_id))

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        body = await request.body()
        return _to_starlette_response(
            await core.chat_completions(
                session_id,
                method=request.method,
                query=request.url.query,
                headers=dict(request.headers),
                body=body,
            )
        )

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        body = await request.body()
        return _to_starlette_response(
            await core.proxy(
                session_id,
                path,
                method=request.method,
                query=request.url.query,
                headers=dict(request.headers),
                body=body,
            )
        )
