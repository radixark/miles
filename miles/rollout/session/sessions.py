"""Single-process FastAPI adapter for the session server.

Thin layer: converts each HTTP request to primitive inputs, calls
``SessionCore``. All session/TITO logic lives in ``core``.
"""

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints.anthropic import utils as anthropic_utils
from sglang.srt.entrypoints.anthropic.serving import convert_response, convert_to_chat_completion_request
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse
from sglang.srt.parser.template_detection import detect_inline_system_support
from starlette.responses import Response

from miles.rollout.session.anthropic_adapter import (
    _ANTHROPIC_ERROR_HEADER_ALLOWLIST as _ANTHROPIC_ERROR_HEADER_ALLOWLIST,
)
from miles.rollout.session.anthropic_adapter import (
    _ANTHROPIC_ERROR_HEADER_PREFIXES as _ANTHROPIC_ERROR_HEADER_PREFIXES,
)
from miles.rollout.session.anthropic_adapter import (
    _anthropic_error_response,
    _anthropic_sse_body,
    _anthropic_wire_json,
    _parse_anthropic_request,
    _restore_anthropic_reasoning_history,
    _strip_anthropic_reasoning_history,
)
from miles.rollout.session.anthropic_adapter import (
    _validate_anthropic_content_block as _validate_anthropic_content_block,
)
from miles.rollout.session.anthropic_adapter import _validate_anthropic_features
from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.core import JSON_MEDIA_TYPE, SessionCore, _render_json
from miles.rollout.session.errors import SessionError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.chat_template_utils.message_matcher_hub import (
    SessionMessageMatcherError,
    resolve_session_message_matcher,
)
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def setup_session_routes(app, backend, config: SessionServerConfig, *, use_addition_r3: bool = False):
    if not config.hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    message_matcher_selector = config.session_message_matcher
    message_matcher = resolve_session_message_matcher(message_matcher_selector)
    logger.info("[session] Using message matcher selector=%r callable=%r", message_matcher_selector, message_matcher)

    tokenizer = load_tokenizer(
        config.hf_checkpoint, chat_template_path=config.chat_template_path, trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=config.tito_model,
        chat_template_kwargs=config.apply_chat_template_kwargs,
    )
    merge_inline_system = not detect_inline_system_support(getattr(tokenizer, "chat_template", None))

    use_v2 = config.use_session_server == "v2"
    if use_v2:
        from miles.rollout.session.v2.core import SessionCoreV2
        from miles.rollout.session.v2.session_state import SessionRegistryV2

        registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito_tokenizer, message_matcher=message_matcher)
        core = SessionCoreV2(backend, registry, config, config.instance_id, use_addition_r3=use_addition_r3)
    else:
        registry = SessionRegistry(tokenizer, tito_tokenizer=tito_tokenizer, message_matcher=message_matcher)
        core = SessionCore(backend, registry, config, config.instance_id, use_addition_r3=use_addition_r3)

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.exception_handler(SessionMessageMatcherError)
    async def session_message_matcher_error_handler(request: Request, exc: SessionMessageMatcherError):
        return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/health")
    async def health():
        response = await core.health()
        body = json.loads(response.body)
        body["anthropic_intermediate_system_supported"] = not merge_inline_system
        return Response(content=_render_json(body), status_code=response.status_code, media_type=JSON_MEDIA_TYPE)

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

    # Keep before session_proxy: Starlette's first match must not bypass session/TITO.
    @app.post("/sessions/{session_id}/v1/messages")
    async def anthropic_messages(request: Request, session_id: str):
        """Serve Anthropic Messages through the OpenAI session path."""
        body = await request.body()
        try:
            anthropic_request = _parse_anthropic_request(body)
            _validate_anthropic_features(anthropic_request)
            try:
                conversion_request, reasoning_history = _strip_anthropic_reasoning_history(anthropic_request)
                openai_request = convert_to_chat_completion_request(
                    conversion_request, merge_inline_system=merge_inline_system
                )
            except Exception as exc:
                logger.exception("Error converting Anthropic request: %s", exc)
                raise ValueError(str(exc)) from exc
            # Core is non-streaming; build fake SSE from its complete response below.
            openai_request.stream = False
            openai_request.stream_options = None
            # Omit defaults so equivalent Anthropic and OpenAI inputs produce the same canonical record.
            openai_body_dict = openai_request.model_dump(
                mode="json", exclude_none=True, exclude_unset=True, by_alias=True
            )
            _restore_anthropic_reasoning_history(openai_body_dict, reasoning_history)
            openai_body = _render_json(openai_body_dict)
        except ValueError as exc:
            # Parsing and JSON encoding failures are invalid Anthropic requests.
            return _anthropic_error_response(400, _render_json({"error": str(exc)}))

        anthropic_stream = bool(anthropic_request.stream)

        try:
            core_response = await core.chat_completions(
                session_id,
                method=request.method,
                query=request.url.query,
                headers=dict(request.headers),
                body=openai_body,
            )
        except SessionError as exc:
            return _anthropic_error_response(exc.status_code, _render_json({"error": str(exc)}))
        except Exception:
            # Preserve Anthropic error framing; cancellation still propagates.
            logger.exception("Anthropic chat processing failed for session %s", session_id)
            return _anthropic_error_response(500, b"")

        if core_response.status_code != 200:
            return _anthropic_error_response(
                core_response.status_code, core_response.body, dict(core_response.headers)
            )

        try:
            openai_response = ChatCompletionResponse.model_validate_json(core_response.body)
            if anthropic_stream:
                events = anthropic_utils.to_anthropic_fake_sse_events(
                    openai_response,
                    model=anthropic_request.model,
                    id_factory=lambda: openai_response.id,
                )
                return Response(
                    content=_anthropic_sse_body(events),
                    status_code=200,
                    headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                    media_type="text/event-stream",
                )
            envelope = convert_response(openai_response).model_copy(update={"id": openai_response.id})
            return Response(content=_anthropic_wire_json(envelope), status_code=200, media_type=JSON_MEDIA_TYPE)
        except Exception:
            # Post-commit failures keep the record and return JSON 500, never partial SSE.
            logger.exception("Anthropic response conversion failed for session %s", session_id)
            return _anthropic_error_response(500, b"")

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
