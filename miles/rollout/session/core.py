"""Logic layer of the session server: ``SessionCore``.

HTTP-agnostic: the FastAPI adapter (``sessions.py`` + ``server.py``) turns each request into primitives and calls these methods. Owns one ``SessionRegistry`` (per-session TITO/trajectory state) and one proxy ``backend``.

- ``chat_completions`` strips the R3 replay payloads (``routed_experts`` / ``indexer_topk``) from the client reply copy-on-write; the ``SessionRecord`` keeps the full response for the training path (``GET /sessions/{id}``).
- ``chat_completions`` holds the per-session lock for prep and state update but not across the proxy call; ``closing`` re-checks and the ``num_assistant`` check gate concurrent DELETE/chat.
- ``stream: true`` is served as fake streaming: the backend call stays non-streaming (TITO needs the complete message + meta_info) and the full response is re-rendered as a single SSE chunk plus ``data: [DONE]``. Errors all happen before the SSE body is built, so they keep their real status codes as JSON.
- ``collect_samples`` assembles training Samples from the session's records on the server (compute -> truncate -> merge, synchronously on the loop like the lock-free ``get_session``); deterministic assembly failures return 422 with the assertion text.
"""

import json
import logging
import time
from dataclasses import dataclass

from starlette.responses import Response

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.errors import SessionNotFoundError, TokenizationError, UpstreamResponseError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.request_contract import PreparedChatRequest, SessionRequestContract
from miles.rollout.session.samples.codec import encode_samples
from miles.rollout.session.samples.merge import (
    compute_samples_from_openai_records,
    merge_samples_with_addition_r3,
    truncate_samples_by_total_tokens,
)
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled

logger = logging.getLogger(__name__)

JSON_MEDIA_TYPE = "application/json"

# Hop-by-hop / length-framing headers dropped from the upstream response so the
# transport layer recomputes them from the body we actually send. "server" and
# "date" are dropped because our own ASGI server always emits them, so echoing
# upstream's copy puts two of each on the wire; aiohttp's parser rejects that
# outright with "Duplicate 'Server' header found" instead of reading the body.
_DROP_RESPONSE_HEADERS = ("content-length", "transfer-encoding", "content-encoding", "server", "date")


@dataclass
class ProxyRequest:
    """Primitive carrier for the proxy backend (replaces fastapi.Request)."""

    method: str
    query: str = ""


def _render_json(payload) -> bytes:
    """Encode like Starlette's JSONResponse (compact, non-ASCII preserved)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _lcp_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token-ID lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _samples_response(payload: bytes) -> Response:
    """The samples-op reply: one safetensors binary payload."""
    return Response(content=payload, status_code=200, media_type="application/octet-stream")


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


def _response_to_stream_chunk(response: dict) -> dict:
    """Synthesize the single ``chat.completion.chunk`` for a fake stream.

    Adapted from NVIDIA-NeMo/ProRL-Agent-Server (``gateway/server.py::_response_to_stream_chunk``)
    and THUDM/slime (``agent/adapters/openai.py::_render_stream``).

    One big delta is protocol-legal (streaming deltas concatenate). All
    tool_calls ride in this one chunk with their ``index`` set: some clients
    mis-assemble arguments fragmented across chunks. The chunk carries no
    ``meta_info``; the training path reads ``GET /sessions/{id}`` instead.
    """
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    delta = {"role": message.get("role", "assistant"), "content": message.get("content")}
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{**tool_call, "index": i} for i, tool_call in enumerate(message["tool_calls"])]
    chunk = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
        "choices": [{"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason")}],
    }
    if response.get("usage") is not None:
        chunk["usage"] = response["usage"]
    return chunk


def _chat_client_response(result: dict, response: dict, client_stream: bool) -> Response:
    if client_stream:
        sse = b"data: " + _render_json(_response_to_stream_chunk(response)) + b"\n\ndata: [DONE]\n\n"
        # Fresh headers: upstream's headers describe its JSON body, not this SSE body.
        # X-Accel-Buffering keeps reverse proxies from buffering the stream.
        return Response(
            content=sse,
            status_code=result["status_code"],
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            media_type="text/event-stream",
        )
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


def extract_completion(result: dict) -> tuple:
    """Decode and validate the backend chat response — shared verbatim by the
    v1 and v2 cores. Returns ``(response, choice, assistant_message,
    completion_token_ids)``; malformed upstream payloads raise
    ``UpstreamResponseError``.
    """
    response = json.loads(result["response_body"])
    choice = response.get("choices", [{}])[0]

    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
        raise UpstreamResponseError("meta_info and output_token_logprobs must be in choice (requires logprobs=True)")
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
    return response, choice, assistant_message, completion_token_ids


class SessionCore:
    """HTTP session operations over one ``SessionRegistry``."""

    def __init__(
        self, backend, registry: SessionRegistry, args, session_server_instance_id=None, *, use_addition_r3=False
    ):
        self.backend = backend
        self.registry = registry
        self.args = args
        self.instance_id = session_server_instance_id
        self.request_contract = SessionRequestContract.from_settings(
            registry.tito_tokenizer,
            force_return_routed_experts=getattr(args, "use_rollout_routing_replay", False),
            force_return_indexer_topk=getattr(args, "use_rollout_indexer_replay", False),
            lora_path=LORA_ADAPTER_NAME if is_lora_enabled(args) else None,
        )
        # Derived from pause_generation_mode at server bootstrap; session code
        # must depend on this capability, never on the weight-update mode.
        self.use_addition_r3 = use_addition_r3

    def _finalize_chat_request(
        self,
        prepared_request: PreparedChatRequest,
        checkpoint_token_ids: list[int],
        prompt_token_ids: list[int],
    ) -> dict:
        """Add session-derived values to the request sent to SGLang.

        ``checkpoint_token_ids`` is the stored snapshot this request builds on
        (v1: the post-rollback checkpoint; v2: the positioned attach node). The
        checkpoint's N - 1 rows must remain a causal prefix of the new prompt,
        so every persisted patch starts exactly where the previous one ended.
        """
        finalize_kwargs = {}
        if self.use_addition_r3 and prepared_request.body.get("return_routed_experts"):
            previous_rows = max(0, len(checkpoint_token_ids) - 1)
            stable_prefix_tokens = _lcp_len(checkpoint_token_ids, prompt_token_ids)
            assert (
                stable_prefix_tokens >= previous_rows
            ), f"additional R3 requires {previous_rows} stable prefix tokens, got {stable_prefix_tokens}"
            finalize_kwargs["routed_experts_start_len"] = previous_rows
        return self.request_contract.finalize(
            prepared_request,
            input_ids=prompt_token_ids,
            **finalize_kwargs,
        )

    async def health(self) -> Response:
        body = {"status": "ok"}
        if self.instance_id is not None:
            body["session_server_instance_id"] = self.instance_id
        return Response(content=_render_json(body), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def create_session(self) -> Response:
        session_id = self.registry.create_session()
        return Response(content=_render_json({"session_id": session_id}), status_code=200, media_type=JSON_MEDIA_TYPE)

    def _session_metadata(self, session_id: str, session) -> dict:
        """The per-session assembly/inspection metadata dict, shared by
        `get_session` (records debug dump) and `collect_samples` (samples op)
        so the two can never drift."""
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
        return metadata

    async def get_session(self, session_id: str) -> Response:
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        payload = GetSessionResponse(session_id=session_id, records=session.records, metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    async def collect_samples(self, session_id: str, *, max_seq_len: int | None) -> Response:
        """Assemble training Samples from this session's records.

        Validation failures return 422; unexpected errors propagate.
        """
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        tokenizer = self.registry.tokenizer
        if not session.records:
            return _samples_response(encode_samples([], metadata, empty_reason="no_records"))
        try:
            samples = compute_samples_from_openai_records(
                self.args,
                session.records,
                tokenizer,
                accumulated_token_ids=metadata.get("accumulated_token_ids"),
                max_trim_tokens=metadata.get("max_trim_tokens", 0),
                use_addition_r3=self.use_addition_r3,
            )
            if max_seq_len is not None:
                samples = truncate_samples_by_total_tokens(samples, max_seq_len, tokenizer)
            if not samples:
                return _samples_response(encode_samples([], metadata, empty_reason="all_truncated"))
            if self.use_addition_r3:
                samples = [merge_samples_with_addition_r3(self.args, samples, session.records, tokenizer)]
            else:
                samples = [merge_samples(samples, tokenizer)]
        except (AssertionError, ValueError) as exc:
            return Response(content=str(exc).encode(), status_code=422, media_type="text/plain")
        return _samples_response(encode_samples(samples, metadata))

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

        Flow: prepare pretokenized input_ids (lock held briefly) → proxy to
        backend (NO lock) → validate response → update trajectory checkpoint and
        append record (lock held briefly). The lock is NOT held during the long
        inference call so DELETE/other ops are not blocked if the agent disconnects.
        """
        request_timestamp = time.time()
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            prepared_request = self.request_contract.prepare(body)
            client_stream = prepared_request.client_stream
            tito_tokenizer = prepared_request.tito_tokenizer

            request_messages = prepared_request.body.get("messages", [])
            prompt_token_ids = session.prepare_pretokenized(
                request_messages,
                tools=prepared_request.body.get("tools"),
                tito_tokenizer=tito_tokenizer,
                message_matcher=self.registry.message_matcher,
            )
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            # prepare_pretokenized applied any retry rollback, so token_ids is
            # the checkpoint this request builds on.
            request_body = self._finalize_chat_request(prepared_request, session.token_ids, prompt_token_ids)

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

        response, _, assistant_message, completion_token_ids = extract_completion(result)

        # --- Phase 3: update state (lock held briefly) ---
        async with session.lock:
            if session.closing:
                logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                return _chat_client_response(result, response, client_stream)

            if session.num_assistant != expected_num_assistant:
                logger.warning(
                    f"Session {session_id} state changed during proxy "
                    f"(expected num_assistant={expected_num_assistant}, "
                    f"got {session.num_assistant}), skipping state update"
                )
                return _chat_client_response(result, response, client_stream)

            session.update_pretokenized_state(
                request_messages,
                assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=self.registry.tito_tokenizer.max_trim_tokens,
            )

            record = SessionRecord(
                timestamp=time.time(),
                request_timestamp=request_timestamp,
                method=method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            session.append_record(record)
        # --- lock released ---

        return _chat_client_response(result, response, client_stream)

    async def proxy(
        self, session_id: str, path: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(
            ProxyRequest(method=method, query=query), path, body=body, headers=headers
        )
        return proxy_result_to_response(result)
