import asyncio
import hashlib
import logging
import time
from copy import deepcopy

import orjson
from starlette.responses import Response

from miles.rollout.session.core import (
    JSON_MEDIA_TYPE,
    ProxyRequest,
    SessionCore,
    _chat_client_response,
    _render_json,
    _samples_response,
    engine_request_body,
    extract_completion,
    proxy_result_to_response,
    store_request_media,
)
from miles.rollout.session.errors import SessionNotFoundError, TokenizationError
from miles.rollout.session.request_args import filter_turn_args, parse_chat_request
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, ROLLOUT_SAMPLING_MASK_FIELDS, encode_samples
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY, build_session_rollout_metrics
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    SessionStateV2,
    commit_generation,
    prepare_token_ids_and_request_args,
)
from miles.rollout.session.v2.utils import build_leaf_material, tree_metadata
from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)


#: Samples builds running in worker threads at the same time, per server. Each
#: holds one session's samples (tokens, images, routed-expert rows) until it
#: returns; more at once trades memory for nothing, the builds are CPU-bound.
SAMPLES_BUILD_SLOTS = 4


class SessionCoreV2(SessionCore):
    """``SessionCore`` with tree serving: overrides the session-semantics
    methods (positioning/commit, metadata, samples op), inherits the
    transport shell (health, create/delete, raw proxy)."""

    def __init__(
        self, backend, registry: SessionRegistryV2, config, session_server_instance_id=None, *, use_addition_r3=False
    ):
        super().__init__(backend, registry, config, session_server_instance_id, use_addition_r3=use_addition_r3)
        # Import-path only in production: function_registry is process-local.
        self.sample_picker = load_function(config.session_sample_picker_path, sync_required=True)
        self.sample_postprocessor = load_function(config.session_sample_postprocessor_path, sync_required=True)
        self._build_slots = asyncio.Semaphore(SAMPLES_BUILD_SLOTS)

    def mismatch_sampled(self, session_id: str) -> bool:
        """Whether this session's samples op computes tito_session_mismatch:
        a per-session draw from a hash of its id against
        ``--tito-session-mismatch-rate`` (0 never, 1 always). The diagnostic
        re-renders the whole conversation, seconds for a 260K-token session,
        and only reports; nothing trains on it."""
        rate = float(getattr(self.config, "tito_session_mismatch_rate", 0.0) or 0.0)
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        draw = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return draw < rate

    def _session_metadata(self, session_id: str, session, *, with_mismatch: bool = True) -> dict:
        """Mirrors ``core.SessionCore._session_metadata``: token ids come from
        the latest committed generation's path, plus the ``tree`` block.
        ``with_mismatch`` False skips the tito_session_mismatch diagnostic
        (the samples op passes ``mismatch_sampled``; the debug read keeps it)."""
        metadata: dict = {}
        mismatch = None
        if with_mismatch:
            try:
                mismatch = self.registry.compute_session_mismatch(session)
            except TokenizationError:
                logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
                mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        latest = session.latest()
        metadata["accumulated_token_ids"] = latest.token_ids if latest is not None else []
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        metadata["tree"] = tree_metadata(session)
        metadata["turn_args"] = filter_turn_args(latest.turn_args) if latest is not None else {}
        return metadata

    async def get_session(self, session_id: str) -> Response:
        """Mirrors ``core.SessionCore.get_session``, serving the records along the
        latest committed generation's path."""
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        latest = session.latest()
        records = [node.record for node in latest.path_nodes()] if latest is not None else []
        payload = GetSessionResponse(session_id=session_id, records=records, metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    async def collect_samples(
        self, session_id: str, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> Response:
        """Samples op: assemble one raw sample per leaf, then run the
        pick/post-process hook pipeline and encode the result.

        The build runs in a worker thread under the session's own lock: the
        lock keeps chat commits of this session out while the tree is read,
        and the event loop keeps serving every other session. On the loop it
        froze a whole server instance for 160 to 190 s per 320-turn session
        (13 sessions, 2026-10-13). At most ``SAMPLES_BUILD_SLOTS`` builds run
        at once: every build holds its session's samples and R3 rows in
        memory, and 32 sessions ending together (the CPU benchmark,
        tests/fast/rollout/test_agentic_rollout_benchmark.py) peaked at
        12.1 GB of tree RSS unbounded against 10.8 GB one at a time.
        Deterministic assembly/hook failures map to 422; unknown exceptions
        propagate.
        """
        session = self.registry.get_session(session_id)
        started = time.perf_counter()
        async with self._build_slots, session.lock:
            worker = asyncio.ensure_future(
                asyncio.to_thread(self._collect_samples_sync, session_id, session, max_seq_len, agent_metadata)
            )
            cancelled = False
            while True:
                try:
                    if cancelled:
                        # Wait without consuming the result: the caller gets
                        # its cancellation, not the build's outcome.
                        await asyncio.wait({worker})
                        break
                    response = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    # The client went away (timeout, disconnect). A thread
                    # cannot be cancelled and keeps reading the tree, so the
                    # lock stays held until it ends, however many cancels
                    # arrive; a chat commit of this session waits.
                    cancelled = True
            if cancelled:
                if worker.done() and not worker.cancelled() and worker.exception() is not None:
                    logger.warning(
                        "samples op for session %s failed after its request was cancelled: %r",
                        session_id,
                        worker.exception(),
                    )
                raise asyncio.CancelledError()
        latest = session.latest()
        logger.info(
            "samples op: session=%s status=%s tokens=%d nodes=%d %.1fs off the event loop",
            session_id,
            response.status_code,
            len(latest.token_ids) if latest is not None else 0,
            len(session.tree.nodes),
            time.perf_counter() - started,
        )
        return response

    def _collect_samples_sync(
        self, session_id: str, session, max_seq_len: int | None, agent_metadata: dict | None
    ) -> Response:
        """The samples build, pure CPU work on the session's tree and media;
        called in a thread with the session lock held."""
        metadata = self._session_metadata(session_id, session, with_mismatch=self.mismatch_sampled(session_id))
        fields = COMPUTED_FIELDS_V2
        if session.sampling_support_replay:
            fields += ROLLOUT_SAMPLING_MASK_FIELDS
        if agent_metadata is not None:
            metadata["agent"] = agent_metadata
        if not session.tree.nodes:
            return _samples_response(encode_samples([], metadata, empty_reason="no_records", fields=fields))

        try:
            material = build_leaf_material(
                self.config,
                session,
                self.registry,
                session_id=session_id,
                max_seq_len=max_seq_len,
                use_addition_r3=self.use_addition_r3,
                # The latest generation is always a leaf; its mismatch is already in the session metadata.
                # A copy: hooks may edit the session metadata, which must not reach that leaf's sample.
                computed_mismatch={session.latest().seq: deepcopy(metadata.get("tito_session_mismatch"))},
            )
        except (AssertionError, ValueError) as exc:
            return Response(content=str(exc).encode(), status_code=422, media_type="text/plain")
        if not material:
            return _samples_response(encode_samples([], metadata, empty_reason="all_truncated", fields=fields))

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
                f"session sample hook failed (picker={self.config.session_sample_picker_path}, "
                f"postprocessor={self.config.session_sample_postprocessor_path}): {exc}"
            )
            return Response(content=body.encode(), status_code=422, media_type="text/plain")
        if not samples:
            return _samples_response(encode_samples([], metadata, empty_reason="all_truncated", fields=fields))
        # Hooks may inspect or mutate session metadata, so publish the
        # authoritative server-owned value only at the wire boundary.
        if self.config.sglang_speculative_algorithm is not None:
            metadata[SESSION_ROLLOUT_METRICS_KEY] = build_session_rollout_metrics(session_id, session.tree.nodes)
        return _samples_response(encode_samples(samples, metadata, fields=fields))

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
        t_prepare = time.perf_counter()
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            client_args = store_request_media(parse_chat_request(body), session.media)
            prepared, attach_parent = prepare_token_ids_and_request_args(
                session,
                client_args,
                config=self.config,
                tito_tokenizer=self.registry.tito_tokenizer,
                message_matcher=self.registry.message_matcher,
            )
            request_body, tito_tokenizer = prepared.body, self.registry.tito_tokenizer
            response_intent = prepared.response_intent
            request_messages = request_body.get("messages", [])
            prompt_token_ids = request_body["input_ids"]
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            checkpoint_token_ids = attach_parent.token_ids if attach_parent is not None else []
            self._maybe_request_addition_r3(request_body, checkpoint_token_ids, prompt_token_ids)
            _add_session_extra_key(request_body, session)

            # orjson: the body carries the whole prompt's input_ids on every turn.
            proxy_body = engine_request_body(request_body, session.media)
        # --- lock released ---

        # --- Phase 2: proxy to backend (NO lock held) ---
        t_proxy = time.perf_counter()
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(
            ProxyRequest(method=method, query=query), "v1/chat/completions", body=proxy_body, headers=headers
        )
        t_commit = time.perf_counter()

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
                logger.debug("Session %s closed during proxy, skipping state update", session_id)
                return _chat_client_response(result, response, response_intent)

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
                turn_args=request_body,
            )
        # --- lock released ---
        if getattr(self.config, "session_chat_timing_log", False):
            done = time.perf_counter()
            logger.info(
                "chat timing: session=%s prepare=%.3fs proxy=%.3fs commit=%.3fs prompt_tokens=%s",
                session_id,
                t_proxy - t_prepare,
                t_commit - t_proxy,
                done - t_commit,
                (response.get("usage") or {}).get("prompt_tokens"),
            )

        return _chat_client_response(result, response, response_intent)


def _add_session_extra_key(request_body: dict, session: SessionStateV2) -> None:
    if session.extra_key is not None:
        request_body["extra_key"] = session.extra_key
