import json
import logging
import math
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_errors import (
    SessionBusyError,
    SessionError,
    SessionInvariantError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.session_types import GetSessionResponse, SessionRecord
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant: {value}")


def _parse_request_body(body: bytes) -> dict:
    return json.loads(body) if body else {}


def _dump_request_body(request_body: dict) -> bytes:
    return json.dumps(request_body).encode()


def _parse_and_validate_response(response_body: bytes) -> tuple[dict, dict, list[int]]:
    """Parse + validate a successful chat completion response.

    Returns (full_response, assistant_message, completion_token_ids). Raises
    UpstreamResponseError on malformed meta_info / content / token-length mismatch.
    Touches no session state.
    """
    try:
        response = json.loads(response_body, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise UpstreamResponseError(f"upstream response is not valid JSON: {e}") from e

    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise UpstreamResponseError("upstream response has no valid choices[0]")
    choice = choices[0]
    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
        raise UpstreamResponseError("meta_info and output_token_logprobs must be in choice (requires logprobs=True)")
    assistant_message = choice.get("message")
    if not isinstance(assistant_message, dict):
        raise UpstreamResponseError("upstream response choice has no valid message")
    if assistant_message.get("content") is None:
        raise UpstreamResponseError(
            "assistant message content is None, when tool call parser failed SGLang should still return "
            "an empty content rather than None. Please check your modified SGLang version."
        )
    output_token_logprobs = meta_info["output_token_logprobs"]
    completion_tokens = meta_info.get("completion_tokens")
    # bool is an int subclass; reject it explicitly so a True/False count is not
    # silently treated as 1/0.
    if (
        not isinstance(output_token_logprobs, list)
        or not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
    ):
        raise UpstreamResponseError("upstream response output_token_logprobs/completion_tokens have invalid types")
    actual_output_logprobs_len = len(output_token_logprobs)
    if actual_output_logprobs_len != completion_tokens:
        raise UpstreamResponseError(
            "invalid chat completion response: "
            f"len(output_token_logprobs)={actual_output_logprobs_len} "
            f"!= completion_tokens={completion_tokens}. "
            f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
        )
    # Each entry must be a (logprob, token_id, ...) sequence (SGLang emits
    # [logprob, token_id, token_text] triples; len > 2 is normal). Both leading
    # fields are consumed downstream: token_id (entry[1]) feeds the stored
    # trajectory token ids, and logprob (entry[0]) feeds Sample.rollout_log_probs
    # in openai_endpoint_utils. A non-int token id or non-numeric logprob would
    # silently corrupt the stored trajectory / training logprobs, so reject the
    # whole response instead of extracting garbage. bool is an int subclass, so
    # reject it explicitly for both fields.
    completion_token_ids: list[int] = []
    for entry in output_token_logprobs:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise UpstreamResponseError(
                "upstream response output_token_logprobs entry is not a (logprob, token_id) pair"
            )
        logprob = entry[0]
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool) or not math.isfinite(logprob):
            raise UpstreamResponseError(
                f"upstream response output_token_logprobs logprob is not a number: {logprob!r}"
            )
        token_id = entry[1]
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise UpstreamResponseError(
                f"upstream response output_token_logprobs token id is not an int: {token_id!r}"
            )
        completion_token_ids.append(token_id)
    return response, assistant_message, completion_token_ids


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

    @app.get("/health")
    async def health():
        body = {"status": "ok"}
        if session_server_instance_id is not None:
            body["session_server_instance_id"] = session_server_instance_id
        return body

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.post("/sessions")
    async def create_session():
        session_id = registry.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = registry.get_session(session_id)
        metadata = {}
        try:
            mismatch = registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = registry.tito_tokenizer.max_trim_tokens
        return GetSessionResponse(
            session_id=session_id,
            records=session.records,
            metadata=metadata,
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        logger.debug(
            f"[session-server] DELETE waiting for lock: session={session_id} lock_locked={session.lock.locked()}"
        )
        await session.lock.acquire()
        logger.debug(f"[session-server] DELETE acquired lock: session={session_id}")
        try:
            registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        """One in-flight chat per session; a second concurrent same-session chat
        fast-fails 409 without entering the backend. State mutation and the
        stateless JSON parse/validation both run inline on the event loop under
        session.lock where state is touched.
        """
        session = registry.get_session(session_id)

        claimed = False
        # claim the single in-flight slot under a brief lock; closing (404) beats busy (409)
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            if session.chat_inflight:
                raise SessionBusyError("session already has an in-flight chat completion")
            session.chat_inflight = True
            claimed = True
        try:
            body = await request.body()
            request_body = _parse_request_body(body)

            # TITO token tracking requires Miles-owned input_ids plus SGLang
            # output-token metadata:
            #   logprobs=True     → populates meta_info.output_token_logprobs
            #   return_meta_info  → wraps the above in choice.meta_info
            # Both flags are hardcoded (not set default) to prevent agent-side
            # overrides from breaking the token accumulation invariants.
            request_body["logprobs"] = True
            request_body["return_meta_info"] = True
            if getattr(args, "use_rollout_routing_replay", False):
                request_body["return_routed_experts"] = True
            if getattr(args, "use_rollout_indexer_replay", False):
                request_body["return_indexer_topk"] = True
            # Must be False so stop-token text is trimmed from assistant
            # message content; token IDs are still taken from logprobs below.
            request_body["no_stop_trim"] = False
            request_messages = request_body.get("messages", [])

            # prepare pretokenized input under the lock (mutates trajectory state)
            async with session.lock:
                if session.closing:
                    raise SessionNotFoundError(f"session not found: session_id={session_id}")
                prompt_token_ids = session.prepare_pretokenized(
                    request_messages,
                    tools=request_body.get("tools"),
                    tito_tokenizer=registry.tito_tokenizer,
                )
                request_body["input_ids"] = prompt_token_ids
                expected_num_assistant = session.num_assistant
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            encoded_body = _dump_request_body(request_body)
            result = await backend.do_proxy(request, "v1/chat/completions", body=encoded_body)
            # Non-200 (e.g. 400 for context too long) passes through unrecorded;
            # the agent can retry or handle the error.
            if result["status_code"] != 200:
                return backend.build_proxy_response(result)

            response, assistant_message, completion_token_ids = _parse_and_validate_response(result["response_body"])

            # commit state under the lock
            async with session.lock:
                if session.closing:
                    logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                    return backend.build_proxy_response(result)
                if session.num_assistant != expected_num_assistant:
                    logger.error(
                        f"Session {session_id} invariant violation: num_assistant={session.num_assistant} "
                        f"!= expected={expected_num_assistant} under the in-flight gate; this should be unreachable"
                    )
                    raise SessionInvariantError(
                        f"session state changed under the in-flight gate (session_id={session_id})"
                    )
                session.update_pretokenized_state(
                    request_messages,
                    assistant_message,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
                )
                record = SessionRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path="/v1/chat/completions",
                    status_code=result["status_code"],
                    request=request_body,
                    response=response,
                )
                session.append_record(record)
            return backend.build_proxy_response(result)
        finally:
            if claimed:
                # single-threaded event loop: a plain write is atomic; no other coroutine
                # mutates this session's flag without the lock, and finally runs on cancellation.
                session.chat_inflight = False

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
