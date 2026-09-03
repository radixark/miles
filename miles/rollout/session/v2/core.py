import json
import logging
import time

from starlette.responses import Response

from miles.rollout.session.core import (
    JSON_MEDIA_TYPE,
    ProxyRequest,
    SessionCore,
    _chat_client_response,
    _render_json,
    _samples_response,
    extract_completion,
    prepare_chat_request,
    proxy_result_to_response,
)
from miles.rollout.session.errors import TokenizationError, reclaimed_error
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, encode_samples
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    commit_generation,
    position_for_request,
    prepare_pretokenized,
)
from miles.rollout.session.v2.utils import build_leaf_material, tree_metadata
from miles.utils.misc import load_function

logger = logging.getLogger(__name__)


class SessionCoreV2(SessionCore):
    """``SessionCore`` with tree serving: overrides the session-semantics
    methods (positioning/commit, metadata, samples op), inherits the
    transport shell (health, create/delete, raw proxy)."""

    def __init__(
        self, backend, registry: SessionRegistryV2, args, session_server_instance_id=None, *, use_addition_r3=False
    ):
        super().__init__(backend, registry, args, session_server_instance_id, use_addition_r3=use_addition_r3)
        # Import-path only in production: function_registry is process-local.
        self.sample_picker = load_function(args.session_sample_picker_path, sync_required=True)
        self.sample_postprocessor = load_function(args.session_sample_postprocessor_path, sync_required=True)

    def _session_metadata(self, session_id: str, session) -> dict:
        """Mirrors ``core.SessionCore._session_metadata``: token ids come from
        the active path, plus the ``tree`` block."""
        metadata: dict = {}
        try:
            mismatch = self.registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.active_token_ids()
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        metadata["tree"] = tree_metadata(session)
        return metadata

    async def get_session(self, session_id: str) -> Response:
        """Mirrors ``core.SessionCore.get_session``, serving ``active_records()``."""
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        payload = GetSessionResponse(session_id=session_id, records=session.active_records(), metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    async def collect_samples(
        self, session_id: str, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> Response:
        """Samples op: assemble one raw sample per leaf, then run the
        pick/post-process hook pipeline and encode the result.

        Synchronous on the server loop (no await), so the session read cannot
        interleave with chat commits. Deterministic assembly/hook failures map
        to 422; unknown exceptions propagate.
        """
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        if agent_metadata is not None:
            metadata["agent"] = agent_metadata
        if not session.tree.nodes:
            return _samples_response(
                encode_samples([], metadata, empty_reason="no_records", fields=COMPUTED_FIELDS_V2)
            )

        try:
            material = build_leaf_material(
                self.args,
                session,
                self.registry,
                session_id=session_id,
                max_seq_len=max_seq_len,
                use_addition_r3=self.use_addition_r3,
            )
        except (AssertionError, ValueError) as exc:
            return Response(content=str(exc).encode(), status_code=422, media_type="text/plain")
        if not material:
            return _samples_response(
                encode_samples([], metadata, empty_reason="all_truncated", fields=COMPUTED_FIELDS_V2)
            )

        # Hook lane: a policy bug is a deterministic 422 carrying the hook's
        # identity, never a masked 500 (server death stays loud).
        try:
            picked = self.sample_picker(material, metadata)
            picked_ids = [id(sample) for sample in picked]
            allowed = {id(sample) for sample in material}
            if any(sample_id not in allowed for sample_id in picked_ids) or len(picked_ids) != len(set(picked_ids)):
                raise ValueError(
                    "pick hook must return a subset of its input samples without duplicates (pure selection)"
                )
            samples = self.sample_postprocessor(picked, metadata)
        except Exception as exc:
            body = (
                f"session sample hook failed (picker={self.args.session_sample_picker_path}, "
                f"postprocessor={self.args.session_sample_postprocessor_path}): {exc}"
            )
            return Response(content=body.encode(), status_code=422, media_type="text/plain")
        if not samples:
            return _samples_response(
                encode_samples([], metadata, empty_reason="all_truncated", fields=COMPUTED_FIELDS_V2)
            )
        return _samples_response(encode_samples(samples, metadata, fields=COMPUTED_FIELDS_V2))

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
            raise reclaimed_error(session_id)

        # --- Phase 1: prepare request (lock held briefly) ---
        async with session.lock:
            if session.closing:
                raise reclaimed_error(session_id)

            request_body, client_stream, tito_tokenizer = prepare_chat_request(
                body, self.args, self.registry.tito_tokenizer
            )

            request_messages = request_body.get("messages", [])
            position_for_request(session, request_messages, message_matcher=self.registry.message_matcher)
            prompt_token_ids = prepare_pretokenized(
                session,
                request_messages,
                tools=request_body.get("tools"),
                tito_tokenizer=tito_tokenizer,
            )
            request_body["input_ids"] = prompt_token_ids
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            self._maybe_request_addition_r3(request_body, session.active_token_ids(), prompt_token_ids)

            proxy_body = json.dumps(request_body).encode()
            attach_parent = session.active_leaf
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

        response, choice, assistant_message, completion_token_ids = extract_completion(result)
        assistant_message = tito_tokenizer.postprocess_completion(
            choice=choice,
            assistant_message=assistant_message,
            completion_token_ids=completion_token_ids,
        )

        # --- Phase 3: update state (lock held briefly) ---
        async with session.lock:
            if session.closing:
                logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                return _chat_client_response(result, response, client_stream)

            record = SessionRecord(
                timestamp=time.time(),
                request_timestamp=request_timestamp,
                method=method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            commit_generation(
                session,
                parent=attach_parent,
                request_messages=request_messages,
                assistant_message=assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=tito_tokenizer.max_trim_tokens,
                record=record,
                response_id=response.get("id", ""),
                finish_reason=choice.get("finish_reason") or "",
            )
        # --- lock released ---

        return _chat_client_response(result, response, client_stream)
