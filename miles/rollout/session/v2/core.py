"""Session server v2 core: tree serving, opt-in via --use-session-server v2.

The v2 twin of ``core.SessionCore`` — same HTTP-facing surface (sessions.py
dispatches on the flag) — but requests position against the trajectory forest
(always-branch serving; --session-strict-append-only fails loud instead) and
the samples op runs the per-leaf assembly + pick/merge hook pipeline with the
extended wire allowlist (rewards + per-sample metadata). Shared HTTP plumbing
(proxy primitives, response rendering) is imported from the v1 module
unchanged; v1 files are not modified.
"""

import inspect
import json
import logging
import time

from starlette.responses import Response

from miles.rollout.session.core import (
    JSON_MEDIA_TYPE,
    ProxyRequest,
    _chat_client_response,
    _render_json,
    _samples_response,
    proxy_result_to_response,
)
from miles.rollout.session.errors import (
    MessageValidationError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, encode_samples_reply
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.rollout.session.v2.assembly import build_leaf_material, default_merge, default_pick, tree_metadata
from miles.rollout.session.v2.session_state import (
    SessionRegistry,
    commit_generation,
    position_for_request,
    prepare_input,
)
from miles.utils.misc import load_function

logger = logging.getLogger(__name__)


class SessionCore:
    """HTTP session operations over one ``SessionRegistry``."""

    def __init__(self, backend, registry: SessionRegistry, args, session_server_instance_id=None):
        self.backend = backend
        self.registry = registry
        self.args = args
        self.instance_id = session_server_instance_id
        # Strict append-only: the single-chain invariant guard (any request
        # that would branch fails loud). Default is always-branch serving.
        self.strict_append_only = bool(getattr(args, "session_strict_append_only", False))
        # The two policy hooks of the samples op (MULTI_LINEAGE_DESIGN v5).
        # Import-path loading only in production (function_registry is
        # process-local and the session server is a separate process); sync
        # callables only — they run on the server event loop.
        self.sample_picker, self.sample_picker_name = self._load_hook(args, "session_sample_picker_path", default_pick)
        self.sample_merger, self.sample_merger_name = self._load_hook(
            args, "session_merge_function_path", default_merge
        )

    @staticmethod
    def _load_hook(args, arg_name: str, default):
        path = getattr(args, arg_name, None)
        if not path:
            return default, f"default:{default.__name__}"
        fn = load_function(path)
        if not callable(fn):
            raise ValueError(f"--{arg_name.replace('_', '-')}={path} did not resolve to a callable")
        if inspect.iscoroutinefunction(fn):
            raise ValueError(
                f"--{arg_name.replace('_', '-')}={path} is async; sample hooks run "
                f"synchronously on the session server event loop"
            )
        return fn, path

    async def health(self) -> Response:
        body = {"status": "ok"}
        if self.instance_id is not None:
            body["session_server_instance_id"] = self.instance_id
        return Response(content=_render_json(body), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def create_session(self) -> Response:
        session_id = self.registry.create_session()
        return Response(content=_render_json({"session_id": session_id}), status_code=200, media_type=JSON_MEDIA_TYPE)

    def _session_metadata(self, session_id: str, state) -> dict:
        """The per-session assembly/inspection metadata dict, shared by
        `get_session` (records debug dump) and `collect_samples` (samples op)
        so the two can never drift."""
        metadata: dict = {}
        try:
            mismatch = self.registry.compute_session_mismatch(state)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = state.active_token_ids()
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        metadata["tree"] = tree_metadata(state)
        return metadata

    async def get_session(self, session_id: str) -> Response:
        state = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, state)
        payload = GetSessionResponse(session_id=session_id, records=state.active_records(), metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    async def collect_samples(
        self, session_id: str, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> Response:
        """Assemble training Samples from this session's records, on the server.

        Runs synchronously on the server loop — no await between reading the
        session state and finishing the reply — the same invariant that makes
        the lock-free `get_session` safe against concurrent chat updates. Do
        not offload the assembly to an executor without snapshotting records
        or holding the session lock.

        Deterministic assembly failures map to 422 with the assertion text as
        the body. They are caught HERE so they never escape
        as an unhandled 500; the ValueError catch also
        covers corrupt stored R3 payloads (binascii/reshape errors) — equally
        deterministic record damage. Unknown exceptions still propagate (a real
        bug must not masquerade as 422).
        """
        state = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, state)
        if agent_metadata is not None:
            metadata["agent"] = agent_metadata
        if not state.tree.nodes:
            return _samples_response(
                encode_samples_reply([], metadata, empty_reason="no_records", fields=COMPUTED_FIELDS_V2)
            )

        def compute_mismatch(messages, token_ids, tools):
            try:
                return self.registry.compute_mismatch(messages, token_ids, tools)
            except TokenizationError:
                logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
                return None

        try:
            material = build_leaf_material(
                self.args,
                state,
                self.registry.tokenizer,
                max_seq_len=max_seq_len,
                max_trim_tokens=self.registry.tito_tokenizer.max_trim_tokens,
                compute_mismatch=compute_mismatch,
            )
        except (AssertionError, ValueError) as exc:
            return Response(content=str(exc).encode(), status_code=422, media_type="text/plain")
        if not material:
            return _samples_response(
                encode_samples_reply([], metadata, empty_reason="all_truncated", fields=COMPUTED_FIELDS_V2)
            )

        # Hook lane: a policy bug is a deterministic 422 carrying the hook's
        # identity, never a masked 500 (server death stays loud).
        try:
            picked = self.sample_picker(material, metadata)
            allowed = {id(sample) for sample in material}
            if any(id(sample) not in allowed for sample in picked):
                raise ValueError("pick hook must return a subset of its input samples (pure selection)")
            samples = self.sample_merger(picked, metadata)
        except Exception as exc:
            body = (
                f"session sample hook failed (picker={self.sample_picker_name}, "
                f"merger={self.sample_merger_name}): {exc}"
            )
            return Response(content=body.encode(), status_code=422, media_type="text/plain")
        if not samples:
            return _samples_response(
                encode_samples_reply([], metadata, empty_reason="all_truncated", fields=COMPUTED_FIELDS_V2)
            )
        return _samples_response(encode_samples_reply(samples, metadata, fields=COMPUTED_FIELDS_V2))

    async def delete_session(self, session_id: str) -> Response:
        state = self.registry.get_session(session_id)
        if state.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        state.closing = True
        # Acquire the lock so an in-flight chat finishes before we drop the session.
        await state.lock.acquire()
        try:
            self.registry.remove_session(session_id)
        finally:
            state.lock.release()
        return Response(status_code=204)

    async def chat_completions(
        self, session_id: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        """Proxy a chat completion through the backend with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → proxy to
        backend (NO lock) → validate response → update trajectory checkpoint and
        append record (lock held briefly). The lock is NOT held during the slow
        proxy call so DELETE/other ops are not blocked if the agent disconnects.
        """
        state = self.registry.get_session(session_id)
        if state.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with state.lock:
            if state.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            try:
                request_body = json.loads(body) if body else {}
            except json.JSONDecodeError as e:
                raise MessageValidationError(f"invalid JSON body: {e}") from e

            # Fake streaming: the backend must stay non-streaming (TITO needs the
            # complete message + meta_info, and sglang rejects return_meta_info
            # with stream=true), so pop the client's intent here and honor it
            # when rendering the client response.
            client_stream = bool(request_body.pop("stream", False))
            request_body.pop("stream_options", None)

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
            position_for_request(state, request_messages, strict=self.strict_append_only)
            prompt_token_ids = prepare_input(
                state,
                request_messages,
                tools=request_body.get("tools"),
                tito_tokenizer=self.registry.tito_tokenizer,
                strict=self.strict_append_only,
            )
            request_body["input_ids"] = prompt_token_ids
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            proxy_body = json.dumps(request_body).encode()
            attach_parent = state.active_leaf
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
        async with state.lock:
            if state.closing:
                logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                return _chat_client_response(result, response, client_stream)

            if self.strict_append_only and state.active_leaf is not attach_parent:
                logger.warning(
                    f"Session {session_id} state changed during proxy "
                    f"(the active view moved), skipping state update"
                )
                return _chat_client_response(result, response, client_stream)

            record = SessionRecord(
                timestamp=time.time(),
                method=method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            commit_generation(
                state,
                parent=attach_parent,
                request_messages=request_messages,
                assistant_message=assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=self.registry.tito_tokenizer.max_trim_tokens,
                record=record,
                response_id=response.get("id", ""),
                finish_reason=choice.get("finish_reason") or "",
            )
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
