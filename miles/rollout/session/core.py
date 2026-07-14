# doc-dev: docs/developer/multi-process-session-server.md
"""Logic layer of the session server: ``SessionCore``.

HTTP-agnostic: the FastAPI adapter (``sessions.py`` + ``server.py``) turns each request into primitives and calls these methods. Owns one ``SessionRegistry`` (per-session TITO/trajectory state) and one proxy ``backend``. Reused unchanged by both chassis — the single-process adapter and the multi-process worker (``worker.py``, primitives decoded off IPC); ``build_session_core`` and ``error_response`` are the single source for core construction and the client error shape so those contracts cannot drift between modes.

- ``create_session(session_id=None)``: the single-process path passes None (the registry mints); the multi-process router mints the id and passes it so the owning worker creates the trajectory under it.
- ``chat_completions`` strips the R3 replay payloads (``routed_experts`` / ``indexer_topk``) from the client reply copy-on-write; the ``SessionRecord`` keeps the full response for the training path (``GET /sessions/{id}``).
- ``chat_completions`` holds the per-session lock for prep and state update but not across the proxy call; ``closing`` re-checks and the ``num_assistant`` check gate concurrent DELETE/chat.
"""

import json
import logging
import time
from dataclasses import dataclass

from starlette.responses import Response

from miles.rollout.session.errors import (
    MessageValidationError,
    SessionError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

JSON_MEDIA_TYPE = "application/json"

# Hop-by-hop / length-framing headers dropped from the upstream response so the
# transport layer recomputes them from the body we actually send.
_DROP_RESPONSE_HEADERS = ("content-length", "transfer-encoding", "content-encoding")


@dataclass
class ProxyRequest:
    """Primitive carrier for the proxy backend (replaces fastapi.Request)."""

    method: str
    query: str = ""


def _render_json(payload) -> bytes:
    """Encode like Starlette's JSONResponse (compact, non-ASCII preserved)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def error_response(exc: SessionError) -> Response:
    """Render a SessionError as the client response."""
    return Response(content=_render_json({"error": str(exc)}), status_code=exc.status_code, media_type=JSON_MEDIA_TYPE)


def build_session_core(backend, args) -> "SessionCore":
    """Construct a SessionCore (tokenizer + registry) from args."""
    tokenizer = load_tokenizer(
        args.hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
        allowed_append_roles=getattr(args, "tito_allowed_append_roles", None),
    )
    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)
    return SessionCore(backend, registry, args, getattr(args, "session_server_instance_id", None))


_CLIENT_STRIPPED_META_KEYS = ("routed_experts", "indexer_topk")


def _strip_replay_payloads(response: dict) -> dict:
    stripped_choices = []
    for choice in response.get("choices", []):
        meta = choice.get("meta_info")
        if isinstance(meta, dict) and any(k in meta for k in _CLIENT_STRIPPED_META_KEYS):
            meta = {k: v for k, v in meta.items() if k not in _CLIENT_STRIPPED_META_KEYS}
            choice = {**choice, "meta_info": meta}
        stripped_choices.append(choice)
    return {**response, "choices": stripped_choices}


def _chat_client_response(result: dict, response: dict) -> Response:
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    return Response(
        content=_render_json(_strip_replay_payloads(response)),
        status_code=result["status_code"],
        headers=headers,
        media_type=JSON_MEDIA_TYPE,
    )


def proxy_result_to_response(result: dict) -> Response:
    """Build the client response from a proxy result.

    Mirrors the previous ``SessionServer.build_proxy_response``: re-emit JSON
    bodies as compact JSON (application/json), pass non-JSON bodies through
    unchanged, and drop wire-level framing headers from upstream.
    """
    content = result["response_body"]
    status_code = result["status_code"]
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    content_type = headers.get("content-type", "")
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Match the old Response(media_type=content_type): pass it through verbatim
        # (incl. "" when upstream sent no content-type) so the wire bytes are identical.
        return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)
    return Response(content=_render_json(data), status_code=status_code, headers=headers, media_type=JSON_MEDIA_TYPE)


class SessionCore:
    """HTTP session operations over one ``SessionRegistry``."""

    def __init__(self, backend, registry: SessionRegistry, args, session_server_instance_id=None):
        self.backend = backend
        self.registry = registry
        self.args = args
        self.instance_id = session_server_instance_id

    async def health(self) -> Response:
        body = {"status": "ok"}
        if self.instance_id is not None:
            body["session_server_instance_id"] = self.instance_id
        return Response(content=_render_json(body), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def create_session(self, session_id: str | None = None) -> Response:
        # None in single-process (the registry mints); the multi-process router
        # mints the id (to route by it) and passes it so the owning worker creates under it.
        session_id = self.registry.create_session(session_id)
        return Response(content=_render_json({"session_id": session_id}), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def get_session(self, session_id: str) -> Response:
        session = self.registry.get_session(session_id)
        metadata: dict = {}
        try:
            mismatch = self.registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        payload = GetSessionResponse(session_id=session_id, records=session.records, metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    async def delete_session(self, session_id: str) -> Response:
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        # Acquire the lock so an in-flight chat finishes before we drop the session.
        await session.lock.acquire()
        try:
            self.registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    async def chat_completions(
        self, session_id: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        """Proxy a chat completion through the backend with TITO token tracking.

        Three phases around the per-session lock: (1) under the lock, parse the
        body, inject the TITO-required request fields, and prepare pretokenized
        ``input_ids`` from the accumulated trajectory; (2) release the lock and
        proxy to the backend — the slow LLM call runs unlocked so DELETE/other
        ops are not blocked if the agent disconnects; (3) re-acquire the lock,
        validate the response, and update the trajectory checkpoint + append the
        record. ``closing`` is re-checked at each lock entry and, together with
        the ``num_assistant`` mismatch check in phase 3, forms the concurrency
        gate that drops a state update whose session was deleted or advanced
        during the unlocked proxy.
        """
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            try:
                request_body = json.loads(body) if body else {}
            except json.JSONDecodeError as e:
                raise MessageValidationError(f"invalid JSON body: {e}") from e

            # TITO token tracking needs Miles-owned input_ids plus SGLang output
            # metadata: logprobs=True populates meta_info.output_token_logprobs and
            # return_meta_info wraps it in choice.meta_info. Hardcoded (not
            # setdefault) so agent-side overrides cannot break token accumulation.
            request_body["logprobs"] = True
            request_body["return_meta_info"] = True
            if getattr(self.args, "use_rollout_routing_replay", False):
                request_body["return_routed_experts"] = True
            if getattr(self.args, "use_rollout_indexer_replay", False):
                request_body["return_indexer_topk"] = True
            # Must be False so stop-token text is trimmed from assistant content;
            # token IDs still come from logprobs below.
            request_body["no_stop_trim"] = False
            # Chat template kwargs should also be forwarded to sglang to make sure
            # parsers work correctly.
            server_ctk = self.registry.tito_tokenizer.chat_template_kwargs
            if server_ctk:
                request_body["chat_template_kwargs"] = {
                    **server_ctk,
                    **(request_body.get("chat_template_kwargs") or {}),
                }

            request_messages = request_body.get("messages", [])
            prompt_token_ids = session.prepare_pretokenized(
                request_messages,
                tools=request_body.get("tools"),
                tito_tokenizer=self.registry.tito_tokenizer,
            )
            request_body["input_ids"] = prompt_token_ids
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            proxy_body = json.dumps(request_body).encode()
            expected_num_assistant = session.num_assistant
        # --- lock released ---

        # --- Phase 2: proxy to backend (NO lock held) ---
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(
            ProxyRequest(method=method, query=query), "v1/chat/completions", body=proxy_body, headers=headers
        )

        # Non-200 (e.g. 400 context too long) passes through unrecorded so the
        # agent can retry or handle the error.
        if result["status_code"] != 200:
            return proxy_result_to_response(result)

        response = json.loads(result["response_body"])
        choice = response.get("choices", [{}])[0]

        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
            raise UpstreamResponseError(
                "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
            )
        assistant_message = choice.get("message") or {}
        if assistant_message.get("content") is None:
            raise UpstreamResponseError(
                "assistant message content is None, when tool call parser failed SGLang should still return "
                "an empty content rather than None. Please check your modified SGLang version."
            )

        output_token_logprobs = meta_info["output_token_logprobs"]
        completion_tokens = meta_info["completion_tokens"]

        actual_output_logprobs_len = len(output_token_logprobs)
        if actual_output_logprobs_len != completion_tokens:
            raise UpstreamResponseError(
                "invalid chat completion response: "
                f"len(output_token_logprobs)={actual_output_logprobs_len} "
                f"!= completion_tokens={completion_tokens}. "
                f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
            )

        completion_token_ids = [t[1] for t in output_token_logprobs]

        # --- Phase 3: update state (lock held briefly) ---
        async with session.lock:
            if session.closing:
                logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                return _chat_client_response(result, response)

            if session.num_assistant != expected_num_assistant:
                logger.warning(
                    f"Session {session_id} state changed during proxy "
                    f"(expected num_assistant={expected_num_assistant}, "
                    f"got {session.num_assistant}), skipping state update"
                )
                return _chat_client_response(result, response)

            session.update_pretokenized_state(
                request_messages,
                assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=self.registry.tito_tokenizer.max_trim_tokens,
            )

            record = SessionRecord(
                timestamp=time.time(),
                method=method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            session.append_record(record)
        # --- lock released ---

        return _chat_client_response(result, response)

    async def proxy(
        self, session_id: str, path: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(
            ProxyRequest(method=method, query=query), path, body=body, headers=headers
        )
        return proxy_result_to_response(result)
