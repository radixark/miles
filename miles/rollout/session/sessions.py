import json
import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_errors import (
    SessionError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.session_types import GetSessionResponse, SessionRecord
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


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
        allowed_append_roles=getattr(args, "tito_allowed_append_roles", None),
    )

    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)

    @app.get("/health")
    async def health():
        body = {"status": "ok"}
        if session_server_instance_id is not None:
            body["session_server_instance_id"] = session_server_instance_id
        return body

    # --- DEBUG: track in-flight chat_completions ---
    _inflight_chat = {"count": 0}

    @app.middleware("http")
    async def debug_request_logger(request: Request, call_next):
        client = request.client
        client_info = f"{client.host}:{client.port}" if client else "unknown"
        logger.info(
            f"[session-server] REQUEST ARRIVED: {request.method} {request.url.path} from={client_info} inflight_chat={_inflight_chat['count']}"
        )
        t0 = time.time()
        response = await call_next(request)
        elapsed = time.time() - t0
        logger.info(
            f"[session-server] REQUEST DONE: {request.method} {request.url.path} status={response.status_code} elapsed={elapsed:.3f}s from={client_info}"
        )
        return response

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
        """Proxy a chat completion through SGLang with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → inject
        SGLang flags → proxy to backend (NO lock) → validate response →
        update trajectory checkpoint (lock held briefly) → append session record.

        The lock is NOT held during the slow proxy call to avoid blocking
        DELETE/other operations when the agent disconnects mid-request.
        """
        _inflight_chat["count"] += 1
        try:
            session = registry.get_session(session_id)
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            # --- Phase 1: prepare request (lock held briefly) ---
            async with session.lock:
                # Double-check: session may have been marked closing while waiting for lock.
                if session.closing:
                    raise SessionNotFoundError(f"session not found: session_id={session_id}")

                body = await request.body()
                request_body = json.loads(body) if body else {}

                # TITO token tracking requires three SGLang flags working together:
                #   logprobs=True            → populates meta_info.output_token_logprobs
                #   return_prompt_token_ids  → adds choice.prompt_token_ids
                #   return_meta_info         → wraps the above in choice.meta_info
                # All three are hardcoded (not setdefault) to prevent agent-side
                # overrides from breaking the token accumulation invariants.
                request_body["logprobs"] = True
                request_body["return_prompt_token_ids"] = True
                request_body["return_meta_info"] = True
                if getattr(args, "use_rollout_routing_replay", False):
                    request_body["return_routed_experts"] = True
                # Must be False so stop tokens are trimmed from output: otherwise the
                # agent sees stop-token text in content, and the accumulated checkpoint
                # would duplicate structural delimiters that the chat template also emits.
                request_body["no_stop_trim"] = False

                request_messages = request_body.get("messages", [])
                pretokenized = session.prepare_pretokenized(
                    request_messages,
                    tools=request_body.get("tools"),
                    tito_tokenizer=registry.tito_tokenizer,
                )
                if pretokenized is not None:
                    request_body["input_ids"] = pretokenized["input_ids"]
                    logger.debug(
                        "Using pretokenized input_ids: %d tokens",
                        len(pretokenized["input_ids"]),
                    )

                body = json.dumps(request_body).encode()
                expected_num_assistant = session.num_assistant
            # --- lock released here ---

            # --- Phase 2: proxy to SGLang (NO lock held) ---
            result = await backend.do_proxy(request, "v1/chat/completions", body=body)

            # If SGLang returned a non-200 error (e.g. 400 for context too long),
            # pass it through to the agent without recording — the agent can retry
            # or handle the error.
            if result["status_code"] != 200:
                return backend.build_proxy_response(result)

            response = json.loads(result["response_body"])

            choice = response.get("choices", [{}])[0]

            meta_info = choice.get("meta_info")
            if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
                raise UpstreamResponseError(
                    "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
                )
            assistant_message = choice.get("message", {})
            if assistant_message.get("content") is None:
                raise UpstreamResponseError(
                    "assistant message content is None, when tool call parser failed SGLang should still return "
                    "an empty content rather than None. Please check your modified SGLang version."
                )

            prompt_token_ids = choice.get("prompt_token_ids")
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
                    return backend.build_proxy_response(result)

                if session.num_assistant != expected_num_assistant:
                    logger.warning(
                        f"Session {session_id} state changed during proxy "
                        f"(expected num_assistant={expected_num_assistant}, "
                        f"got {session.num_assistant}), skipping state update"
                    )
                    return backend.build_proxy_response(result)

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
            # --- lock released here ---

            return backend.build_proxy_response(result)
        finally:
            _inflight_chat["count"] -= 1

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
