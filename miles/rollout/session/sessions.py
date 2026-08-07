"""Single-process FastAPI adapter for the session server.

Thin layer: converts each HTTP request to primitive inputs, calls
``SessionCore``. All session/TITO logic lives in ``core``.
"""

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import SessionError, SessionMessageMatcherError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.chat_template_utils.message_matcher_hub import SessionMessageMatcher, resolve_session_message_matcher
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def _validated_message_matcher(matcher: SessionMessageMatcher) -> SessionMessageMatcher:
    """Enforce the matcher contract at runtime: exact bool, no exceptions.

    A truthy non-bool or a raised exception is a configuration error, never a
    match or mismatch; surface it as ``SessionMessageMatcherError`` (500)
    instead of silently deciding session identity from a broken matcher.
    """

    def validated(stored: dict, replayed: dict) -> bool:
        try:
            result = matcher(stored, replayed)
        except Exception as exc:
            raise SessionMessageMatcherError("session message matcher raised an exception") from exc
        if type(result) is not bool:
            raise SessionMessageMatcherError(f"session message matcher must return bool, got {type(result).__name__}")
        return result

    return validated


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    session_server_instance_id = getattr(args, "session_server_instance_id", None)

    message_matcher_selector = getattr(args, "session_message_matcher", "strict")
    resolved_matcher = resolve_session_message_matcher(message_matcher_selector)
    logger.info(
        "[session] Using message matcher selector=%r callable=%s.%s",
        message_matcher_selector,
        getattr(resolved_matcher, "__module__", "<unknown>"),
        getattr(resolved_matcher, "__qualname__", type(resolved_matcher).__name__),
    )
    message_matcher = _validated_message_matcher(resolved_matcher)

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
    )

    use_v2 = getattr(args, "use_session_server", None) == "v2"
    if use_v2:
        from miles.rollout.session.v2.core import SessionCoreV2
        from miles.rollout.session.v2.session_state import SessionRegistryV2

        registry = SessionRegistryV2(
            args,
            tokenizer,
            tito_tokenizer=tito_tokenizer,
            message_matcher=message_matcher,
            message_matcher_selector=message_matcher_selector,
        )
        core = SessionCoreV2(backend, registry, args, session_server_instance_id)
    else:
        registry = SessionRegistry(
            args,
            tokenizer,
            tito_tokenizer=tito_tokenizer,
            message_matcher=message_matcher,
            message_matcher_selector=message_matcher_selector,
        )
        core = SessionCore(backend, registry, args, session_server_instance_id)

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

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

    @app.post("/sessions/{session_id}/samples")
    async def collect_samples(request: Request, session_id: str):
        # Starlette matches routes in registration order; keep this before session_proxy.
        # Parse here so malformed input is not reported as an assembly error (422).
        body = await request.body()
        params = json.loads(body) if body else {}
        if use_v2:
            return await core.collect_samples(
                session_id, max_seq_len=params.get("max_seq_len"), agent_metadata=params.get("metadata")
            )
        return await core.collect_samples(session_id, max_seq_len=params.get("max_seq_len"))

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
