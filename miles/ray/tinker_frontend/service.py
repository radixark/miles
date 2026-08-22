"""The tinker frontend service: official SDK verbs -> backend operations.

Request -> ordinal mapping (the D5 note in operations.py): the 0.24.1 SDK
holds one per-model counter — every training verb (each forward_backward
chunk, forward chunk, optim_step, save/load, sampler publish) consumes one
``seq_id``, consecutive from 1 — which is exactly the backend ledger's
per-registration ordinal contract. The frontend therefore forwards
``ordinal = seq_id`` verbatim; chunks the SDK posts out of order (first
chunk last, by design) arrive out of order and the ledger gap-buffers them.
A submission this layer rejects still consumes its ordinal as a terminal
FAILED(user) ledger record, so one bad chunk can never leave a gap that
starves the registration.

Future protocol: every heavy verb returns ``{"request_id"}`` and the SDK
polls /api/v1/retrieve_future. Request ids are deterministic in the SDK's
own coordinates ((session, model_seq_id) / (model, seq_id)), so a resent
submission lands on its original record: identical -> replay, different ->
422 (the SDK treats 409 as retryable, so a real conflict must never be 409).
Terminal bodies are stored for replay BEFORE the backend record is acked —
a response lost on the wire is re-polled and must find the same bytes.

This layer is deliberately thin: datum/loss validation and translation live
in translation.py, execution semantics live behind the controller surface
(register/deregister/enqueue/reject/get/ack + registry state), and sampling
proxies to the sglang router under the registration-scoped serving name.

Sampling is additionally guarded by a context preflight (prompt + max_tokens
against the engine context limit — configured or discovered, typed 400
before identity consumption) and observed through SamplingAdmission/
SamplingStats counters; a background maintenance loop reaps orphaned
sessions and futures without ever freeing an identity (code-0815 §6/§7).
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from miles.ray.multi_lora.config import AdapterRunConfig
from miles.ray.multi_lora.identity import cache_extra_key, make_rid, serving_lora_name
from miles.ray.multi_lora.operations import OperationBackpressure
from miles.ray.tinker_frontend import translation, wire
from miles.ray.tinker_frontend.sampling import SamplingTransport, SGLangRouterSamplingTransport
from miles.ray.tinker_frontend.state import (
    CheckpointCatalog,
    CheckpointRecord,
    ConflictError,
    ExpiredError,
    FutureRecord,
    FutureStore,
    ModelRecord,
    ModelStore,
    SamplingSessionRecord,
    SamplingSessionStore,
    SessionStore,
    fingerprint_of,
)
from miles.ray.tinker_frontend.translation import UserInputError

logger = logging.getLogger(__name__)

_LEDGER_CONFLICT_MARKS = ("different content", "already taken")


class ApiError(Exception):
    """Maps to an HTTP error response (submit-time failures the SDK should
    see as a status code, not a terminal future)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class SamplingAdmission:
    """Global fail-fast sampling admission, counted in sub-generations: a
    logical request weighs ``num_samples`` because each sample fans out into
    its own router call.

    The SDK's per-client ``sample_max_concurrent_requests=64`` bounds ONE
    client; the aggregate across clients was unbounded, and >100 concurrent
    generations hit the shared router client's implicit 100-connection/10s
    pool deadline as empty terminal failures the SDK never retries (the Tau
    sampling cliff). Rejecting here — BEFORE the request consumes its seq
    identity or mints a FutureRecord — maps to HTTP 429 + Retry-After, which
    the SDK retries with backoff using the SAME seq id, so an admitted
    request still executes exactly once. Single event loop, no awaits
    between check and acquire: admission is atomic with submission."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.in_use = 0
        self.rejected = 0  # total backpressured submissions (429s)
        self.admitted = 0  # admitted logical requests
        self.admitted_weight = 0  # admitted sub-generations (sum of weights)
        self.peak_in_use = 0  # high-water of concurrently active sub-generations

    def try_acquire(self, weight: int) -> bool:
        if self.in_use + weight > self.capacity:
            self.rejected += 1
            return False
        self.in_use += weight
        self.admitted += 1
        self.admitted_weight += weight
        if self.in_use > self.peak_in_use:
            self.peak_in_use = self.in_use
        return True

    def release(self, weight: int) -> None:
        self.in_use -= weight


class SamplingStats:
    """Aggregate sampling terminal counters (the code-0815 §6.1 minimal set;
    admission-side counts live on SamplingAdmission). Latencies are per
    logical request: submit -> first completed sub-generation (the closest
    observable to time-to-first-token over a non-streaming router hop) and
    submit -> terminal."""

    def __init__(self) -> None:
        self.completed = 0
        self.failed = 0
        self.failures_by_class: dict[str, int] = {}
        self.first_result_s_sum = 0.0
        self.first_result_s_max = 0.0
        self.first_result_count = 0
        self.total_s_sum = 0.0
        self.total_s_max = 0.0

    def record_latency(self, first_result_s: float | None, total_s: float) -> None:
        self.total_s_sum += total_s
        self.total_s_max = max(self.total_s_max, total_s)
        if first_result_s is not None:
            self.first_result_s_sum += first_result_s
            self.first_result_s_max = max(self.first_result_s_max, first_result_s)
            self.first_result_count += 1

    def record_failure(self, failure_class: str) -> None:
        self.failed += 1
        self.failures_by_class[failure_class] = self.failures_by_class.get(failure_class, 0) + 1


# The engine context limit out of sglang's /get_server_info. The response is
# ``{**asdict(ServerArgs), **scheduler_info, ...}``: ``context_length`` echoes
# an explicitly configured limit (null when derived from the model config),
# and the scheduler always reports ``max_req_input_len``, which it computes as
# ``min(context_len - 1, kv_pool_tokens - 1) - 5`` — so ``+ 6`` reconstructs
# the effective context (folding in the KV-pool bound when that is tighter).
def _context_limit_from_server_info(info: Any) -> int | None:
    if not isinstance(info, dict):
        return None
    limits = []
    context_length = info.get("context_length")
    if isinstance(context_length, int) and not isinstance(context_length, bool) and context_length > 0:
        limits.append(context_length)
    max_req_input_len = info.get("max_req_input_len")
    if isinstance(max_req_input_len, int) and not isinstance(max_req_input_len, bool) and max_req_input_len > 0:
        limits.append(max_req_input_len + 6)
    return min(limits, default=None)


def _note_first_result(task: asyncio.Task, record: "FutureRecord") -> None:
    """Done-callback on each sub-generation: stamps when the request's FIRST
    sub-generation finished (queue-to-first-result latency). Cancellations
    are not results."""
    if not task.cancelled() and record.first_result_at is None:
        record.first_result_at = time.time()


class TinkerFrontend:
    """One instance per controller; single event loop, no cross-await state
    mutation inside a submit or resolve step."""

    def __init__(
        self,
        backend: Any,
        poll_window_s: float = 15.0,
        poll_interval_s: float = 0.1,
        sampling_transport: SamplingTransport | None = None,
        sampling_max_active_subgenerations: int = 64,
        sampling_max_context: int | None = None,
        session_idle_ttl_s: float = 3600.0,
        future_unpolled_ttl_s: float = 900.0,
        future_undelivered_ttl_s: float = 3600.0,
        maintenance_interval_s: float = 15.0,
    ) -> None:
        self.backend = backend
        self.poll_window_s = poll_window_s
        self.poll_interval_s = poll_interval_s
        # One capacity, two layers: fail-fast admission at submit (429 before
        # identity consumption) and the transport's hard in-flight bound
        # (last-resort invariant). 64 is the GPU-validated safe default, not
        # a universal optimum — deployments tune it via
        # --tinker-sampling-max-active-subgenerations.
        self.sampling_admission = SamplingAdmission(sampling_max_active_subgenerations)
        self.sampling_stats = SamplingStats()
        # Engine context limit for the sampling preflight (prompt + max_tokens
        # must fit): statically configured here, or discovered lazily from the
        # transport's server_info on the first sample. None = not yet known;
        # the preflight only ever rejects against a KNOWN limit.
        self._context_limit = sampling_max_context
        self._context_limit_source = "configured" if sampling_max_context is not None else None
        self._context_discovery_task: asyncio.Task | None = None
        self._context_discovery_attempts = 0
        # Orphan reaping TTLs (<= 0 disables that class of reaping).
        self.session_idle_ttl_s = session_idle_ttl_s
        self.future_unpolled_ttl_s = future_unpolled_ttl_s
        self.future_undelivered_ttl_s = future_undelivered_ttl_s
        self.maintenance_interval_s = maintenance_interval_s
        self._maintenance_task: asyncio.Task | None = None
        self._stats_logged: tuple | None = None
        # Injected sampling hop (frontend -> router); the default preserves
        # the direct-router transport this frontend always used.
        self.sampling_transport = (
            sampling_transport
            if sampling_transport is not None
            else SGLangRouterSamplingTransport(
                backend.sampling_endpoint(), max_inflight=sampling_max_active_subgenerations
            )
        )
        self.sessions = SessionStore()
        self.models = ModelStore()
        self.futures = FutureStore()
        self.checkpoints = CheckpointCatalog()
        self.samplers = SamplingSessionStore()
        self._sample_tasks: set[asyncio.Task] = set()
        # request_id -> task, so the reaper can cancel one orphaned sample.
        self._sample_task_by_request: dict[str, asyncio.Task] = {}
        self._closing = False

    async def close(self) -> None:
        """Idempotent shutdown barrier: gate new samples, stop the background
        maintenance/discovery tasks, cancel AND await every in-flight sample
        task (so the transport observes cancellation before it is closed
        under it), then close the transport."""
        self._closing = True
        background = [task for task in (self._maintenance_task, self._context_discovery_task) if task is not None]
        self._maintenance_task = None
        self._context_discovery_task = None
        for task in background:
            task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        tasks = list(self._sample_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            # The done-callbacks discard too, but only on a later loop tick;
            # close() must return with the set verifiably drained.
            self._sample_tasks.difference_update(tasks)
        self._sample_task_by_request.clear()
        await self.sampling_transport.close()

    # ---------------- maintenance: orphan reaping + metrics summary ----------------

    def start_maintenance(self) -> None:
        """Start the background maintenance loop (idempotent). Owned by the
        HTTP server's start — a frontend embedded in tests drives reap_once
        directly with an injected clock instead."""
        if self._maintenance_task is None and not self._closing:
            self._maintenance_task = asyncio.get_running_loop().create_task(self._maintenance_loop())

    async def _maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(self.maintenance_interval_s)
            try:
                self.reap_once()
                self._log_sampling_summary()
            except Exception:  # noqa: BLE001 — maintenance must never die silently mid-run
                logger.exception("[tinker] frontend maintenance tick failed")

    def reap_once(self, now: float | None = None) -> dict[str, int]:
        """One reaping pass (code-0815 §7), replay-idempotency preserved by
        construction — reaping frees bytes and capacity without permitting
        re-execution:

        - idle sessions (no heartbeat past the TTL): the session record and
          its sampling sessions go together; old sampler ids fail closed;
        - orphaned sample futures (client stopped polling past the TTL): the
          server-side generation is cancelled (releasing admission permits
          and transport slots via the existing done-callbacks) and the future
          resolves typed with the reap reason — the seq was spent at submit
          and stays spent;
        - unpolled operation-family futures past the same TTL: polled once on
          the client's behalf, which stores the terminal bytes BEFORE acking
          the ledger record (the existing ack-based retention order), so the
          unacked-results budget drains for vanished clients;
        - terminal futures never retrieved past the undelivered TTL: evicted
          to a fingerprint tombstone — a late retry gets a typed 410, never a
          re-execution.
        """
        now = time.time() if now is None else now
        counts = {"sessions": 0, "cancelled_samples": 0, "undelivered": 0}
        if self.session_idle_ttl_s > 0:
            idle_sessions = self.sessions.reap_idle(self.session_idle_ttl_s, now)
            self.samplers.remove_for_sessions({session.session_id for session in idle_sessions})
            for session in idle_sessions:
                counts["sessions"] += 1
                logger.info(
                    f"[tinker] reaped idle session '{session.session_id}' (no heartbeat for "
                    f"{now - session.last_heartbeat:.0f}s; its sampling sessions were retired)"
                )
        for record in list(self.futures.records.values()):
            if record.terminal is None:
                if self.future_unpolled_ttl_s <= 0:
                    continue
                idle_s = now - max(record.created_at, record.last_polled_at)
                if idle_s <= self.future_unpolled_ttl_s:
                    continue
                if record.kind == "sample":
                    task = self._sample_task_by_request.get(record.request_id)
                    if task is not None and not task.done():
                        record.cancel_reason = (
                            f"sampling request '{record.request_id}' was orphaned (not polled for "
                            f"{idle_s:.0f}s) and its generation was cancelled by the reaper; the seq "
                            "identity stays spent — resubmitting it will not re-run the generation"
                        )
                        task.cancel()
                        counts["cancelled_samples"] += 1
                        logger.warning(
                            f"[tinker] reaped orphaned sample '{record.request_id}': cancelled its "
                            f"generation after {idle_s:.0f}s without a poll"
                        )
                else:
                    # The training/lifecycle ledger owns execution — never
                    # cancel it. Resolving on the vanished client's behalf
                    # moves the terminal bytes here and acks the ledger.
                    self._poll(record)
            elif self.future_undelivered_ttl_s > 0 and not self.futures.is_delivered(record.request_id):
                age_s = now - max(record.resolved_at or record.created_at, record.last_polled_at)
                if age_s > self.future_undelivered_ttl_s:
                    self.futures.reap_undelivered(record)
                    counts["undelivered"] += 1
                    logger.info(
                        f"[tinker] reaped undelivered terminal future '{record.request_id}' "
                        f"({record.kind}, unretrieved for {age_s:.0f}s); a tombstone keeps its identity"
                    )
        return counts

    def _log_sampling_summary(self) -> None:
        """Periodic aggregate line (INFO), only when something changed since
        the last tick — the per-request lines are DEBUG/WARNING."""
        admission, stats = self.sampling_admission, self.sampling_stats
        snapshot = (admission.admitted, admission.rejected, stats.completed, stats.failed)
        if snapshot == self._stats_logged:
            return
        self._stats_logged = snapshot
        finished = stats.completed + stats.failed
        mean_total = stats.total_s_sum / finished if finished else 0.0
        mean_first = stats.first_result_s_sum / stats.first_result_count if stats.first_result_count else 0.0
        logger.info(
            f"[tinker] sampling summary: admitted={admission.admitted} ({admission.admitted_weight} "
            f"sub-generations) rejected_429={admission.rejected} active={admission.in_use}"
            f"/{admission.capacity} peak={admission.peak_in_use} completed={stats.completed} "
            f"failed={stats.failed} failures_by_class={stats.failures_by_class} "
            f"queue_to_first_result_s(mean/max)={mean_first:.3f}/{stats.first_result_s_max:.3f} "
            f"total_s(mean/max)={mean_total:.3f}/{stats.total_s_max:.3f} "
            f"context_limit={self._context_limit}"
        )

    # ---------------- bootstrap ----------------

    def health(self) -> dict:
        # Readiness, not liveness (/health): the socket accepting connections
        # says nothing about the trainer, which starts later and can fail.
        if not getattr(self.backend, "trainer_ready", True):
            raise ApiError(503, "trainer is initializing; the service is not ready for SDK traffic yet")
        return {"status": "ok"}

    def _check_sdk_version(self, sdk_version: str) -> None:
        # Exact pin: this frontend mirrors the request shapes tinker==0.24.1
        # actually POSTs. A different patch of 0.24.x is untested wire surface
        # (and 0.25+ switches forward_backward to protobuf mid-run) — reject
        # at bootstrap, where the version travels with the request.
        if sdk_version != wire.TINKER_SDK_VERSION_PIN:
            raise ApiError(
                400,
                f"unsupported tinker SDK version '{sdk_version}': this deployment serves exactly "
                f"tinker=={wire.TINKER_SDK_VERSION_PIN}. Pin tinker=={wire.TINKER_SDK_VERSION_PIN}.",
            )

    def client_config(self, request: wire.ClientConfigRequest) -> dict:
        self._check_sdk_version(request.sdk_version)
        return dict(wire.CLIENT_CONFIG_FLAGS)

    def capabilities(self) -> dict:
        info = self.backend.service_info()
        # None until the engine context limit is configured or discovered.
        model = {"model_name": info.get("base_model"), "max_context_length": self._context_limit}
        return {"supported_models": [model]}

    def create_session(self, request: wire.CreateSessionRequest) -> dict:
        self._check_sdk_version(request.sdk_version)
        record = self.sessions.create(request.sdk_version, request.tags, request.user_metadata)
        return {"type": "create_session", "session_id": record.session_id}

    def session_heartbeat(self, request: wire.SessionHeartbeatRequest) -> dict:
        if not self.sessions.heartbeat(request.session_id):
            raise ApiError(404, f"unknown session '{request.session_id}'")
        return {"type": "session_heartbeat"}

    def telemetry(self, _body: Any) -> dict:
        return {"status": "accepted"}

    # ---------------- models ----------------

    def _base_model(self) -> str:
        return self.backend.service_info().get("base_model") or ""

    def _model_for(self, model_id: str | None) -> ModelRecord:
        model = self.models.get(model_id) if model_id else None
        if model is None:
            raise ApiError(404, f"unknown model_id '{model_id}'")
        return model

    async def create_model(self, request: wire.CreateModelRequest) -> dict:
        session = self.sessions.get(request.session_id)
        if session is None:
            raise ApiError(404, f"unknown session '{request.session_id}'")
        fingerprint = fingerprint_of(request.model_dump(mode="json"))
        name = f"t{session.short}-m{request.model_seq_id}"
        request_id = f"{name}:create"
        if (existing := self._existing(request_id, fingerprint)) is not None:
            return wire.untyped_future(request_id, existing.model.model_id if existing.model else None)

        lora = request.lora_config
        if lora is None:
            raise ApiError(400, "lora_config is required: this deployment serves LoRA training runs only")
        if lora.seed is not None:
            raise ApiError(400, "lora_config.seed cannot be honored by this deployment; omit it")
        if not (lora.train_unembed and lora.train_mlp and lora.train_attn):
            raise ApiError(
                400,
                "per-module train flags cannot be honored: trained modules are deployment-wide "
                "(--target-modules); leave train_unembed/train_mlp/train_attn at their defaults",
            )
        base_model = self._base_model()
        if request.base_model != base_model:
            raise ApiError(
                400, f"base_model '{request.base_model}' is not served; this deployment serves '{base_model}'"
            )

        metadata = {"session_id": request.session_id, "model_seq_id": request.model_seq_id}
        if request.user_metadata:
            metadata["user_metadata"] = request.user_metadata
        try:
            await self.backend.register(name, AdapterRunConfig(rank=lora.rank, metadata=metadata))
        except ValueError as exc:
            # A concurrent identical create may have raced this one.
            if (existing := self._existing(request_id, fingerprint)) is not None:
                return wire.untyped_future(request_id, existing.model.model_id if existing.model else None)
            raise ApiError(400, str(exc)) from exc
        registered = self.backend.registration_view(name)
        model = ModelRecord(
            model_id=f"{request.session_id}:train:{request.model_seq_id}",
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            name=name,
            registration_id=registered["registration_id"],
            base_model=base_model,
            rank=registered["rank"],
            fingerprint=fingerprint,
        )
        self.models.add(model)
        self.futures.put(
            FutureRecord(request_id=request_id, kind="create_model", fingerprint=fingerprint, model=model)
        )
        return wire.untyped_future(request_id, model.model_id)

    def get_info(self, request: wire.GetInfoRequest) -> dict:
        model = self._model_for(request.model_id)
        return {
            "type": "get_info",
            "model_id": model.model_id,
            "model_data": {"arch": None, "model_name": model.base_model, "tokenizer_id": model.base_model},
            "is_lora": True,
            "lora_rank": model.rank,
            "model_name": model.base_model,
        }

    async def unload_model(self, request: wire.UnloadModelRequest) -> dict:
        model = self._model_for(request.model_id)
        fingerprint = fingerprint_of(request.model_dump(mode="json"))
        request_id = f"{model.name}.{model.rid8}:unload"
        if self._existing(request_id, fingerprint) is not None:
            return wire.untyped_future(request_id, model.model_id)
        # Registration-pinned: a same-name successor must never be retired
        # by a stale handle (the backend re-checks under the same pin).
        await self.backend.deregister(model.name, model.registration_id)
        self.futures.put(
            FutureRecord(request_id=request_id, kind="unload_model", fingerprint=fingerprint, model=model)
        )
        return wire.untyped_future(request_id, model.model_id)

    # ---------------- training operations ----------------

    def forward_backward(self, request: wire.ForwardBackwardRequest) -> dict:
        return self._submit_operation(
            request,
            request.model_id,
            request.seq_id,
            "forward_backward",
            lambda: translation.fb_input_to_payload(request.forward_backward_input),
        )

    def forward(self, request: wire.ForwardRequest) -> dict:
        def prepare(record: FutureRecord, payload: dict) -> None:
            # The backend attaches loss metrics to forward_backward results
            # only; keep the request payload for the forward recompute.
            record.forward_payload = payload

        return self._submit_operation(
            request,
            request.model_id,
            request.seq_id,
            "forward",
            lambda: translation.fb_input_to_payload(request.forward_input),
            prepare=prepare,
        )

    def optim_step(self, request: wire.OptimStepRequest) -> dict:
        return self._submit_operation(
            request,
            request.model_id,
            request.seq_id,
            "optim_step",
            lambda: translation.adam_params_to_payload(request.adam_params),
        )

    def save_weights(self, request: wire.SaveWeightsRequest) -> dict:
        def build() -> dict:
            if request.overwrite:
                raise UserInputError("overwrite=true is not supported: named states are immutable")
            if request.ttl_seconds is not None:
                # No reaper runs in v1: accepting a TTL would promise an expiry
                # that never happens. Same typed rejection as sampler publishes.
                raise UserInputError("ttl_seconds is not supported in v1 (checkpoints never expire); omit it")
            payload: dict = {}
            if request.path is not None:
                payload["tag"] = request.path
            return payload

        return self._submit_operation(request, request.model_id, request.seq_id, "save_state", build)

    def load_weights(self, request: wire.LoadWeightsRequest) -> dict:
        if request.model_id is None:
            # create_model_via_load_weights is advertised off; the SDK only
            # sends session addressing when the server enables that flag.
            raise ApiError(400, "load_weights requires model_id (session-addressed creation is not supported)")

        def build() -> dict:
            if not request.optimizer:
                raise UserInputError(
                    "weights-only restore is not supported in v1 (the backend restores the full training "
                    "state); use load_state_with_optimizer / create_training_client_from_state_with_optimizer"
                )
            checkpoint = self.checkpoints.get(request.path)
            if checkpoint is None:
                raise UserInputError(
                    f"unknown checkpoint '{request.path}'; v1 resolves paths minted during this service lifetime"
                )
            return {"path": checkpoint.backend_path}

        def prepare(record: FutureRecord, payload: dict) -> None:
            record.tinker_path = request.path
            # Redaction: failures echo the trainer-side path; swap it back for
            # the public URI before the error body reaches the client.
            record.backend_target = {"path": payload["path"]}

        return self._submit_operation(request, request.model_id, request.seq_id, "load_state", build, prepare=prepare)

    def save_weights_for_sampler(self, request: wire.SaveWeightsForSamplerRequest) -> dict:
        model = self._model_for(request.model_id)

        def build() -> dict:
            if self.sessions.get(model.session_id) is None:
                raise UserInputError("the parent session expired; create a new session before publishing a sampler")
            if request.path is not None:
                raise UserInputError(
                    "named sampler checkpoints are not supported in v1 (latest-only serving); use "
                    "save_weights_and_get_sampling_client for ephemeral sampling"
                )
            if request.sampling_session_seq_id is None:
                raise UserInputError("save_weights_for_sampler without a path needs sampling_session_seq_id")
            if request.ttl_seconds is not None:
                raise UserInputError("ttl_seconds is not supported for sampler publishes in v1")
            return {}

        def prepare(record: FutureRecord, payload: dict) -> None:
            session = self.sessions.get(model.session_id)
            short = session.short if session is not None else model.session_id[:12]
            record.sampling_session_id = f"samp-{short}-ss{request.sampling_session_seq_id}"

        # The official 0.24.1 client increments its sampling counter INSIDE
        # the HTTP retry closure (training_client.py: _send_request mints a
        # fresh sampling_session_seq_id per attempt) while the operation
        # seq_id stays fixed. A response lost on the wire therefore retries
        # the SAME operation identity with a different sampling sequence —
        # fingerprinting that field would turn the retry into a fatal 422.
        # The operation seq_id remains authoritative; replay returns the
        # originally minted sampler id.
        fingerprint_dump = request.model_dump(mode="json")
        fingerprint_dump.pop("sampling_session_seq_id", None)
        return self._submit_operation(
            request,
            request.model_id,
            request.seq_id,
            "save_weights_for_sampler",
            build,
            prepare=prepare,
            fingerprint_dump=fingerprint_dump,
        )

    def _existing(self, request_id: str, fingerprint: str) -> FutureRecord | None:
        try:
            return self.futures.existing(request_id, fingerprint)
        except ExpiredError as exc:
            raise ApiError(410, str(exc)) from exc
        except ConflictError as exc:
            raise ApiError(422, str(exc)) from exc

    def _submit_operation(
        self,
        request: wire.WireModel,
        model_id: str | None,
        seq_id: int | None,
        kind: str,
        build_payload: Callable[[], dict],
        prepare: Callable[[FutureRecord, dict], None] | None = None,
        fingerprint_dump: dict | None = None,
    ) -> dict:
        model = self._model_for(model_id)
        if seq_id is None or seq_id < 1:
            raise ApiError(400, f"{kind} needs a seq_id >= 1")
        request_dump = request.model_dump(mode="json")
        # ``fingerprint_dump`` lets a verb exclude fields the official SDK
        # regenerates per retry attempt (save_weights_for_sampler's
        # sampling_session_seq_id) from the retry-identity fingerprint.
        fingerprint = fingerprint_of(fingerprint_dump if fingerprint_dump is not None else request_dump)
        request_id = f"{model.name}.{model.rid8}:op{seq_id}"
        if self._existing(request_id, fingerprint) is not None:
            return wire.untyped_future(request_id, model.model_id)

        record = FutureRecord(
            request_id=request_id,
            kind="operation",
            fingerprint=fingerprint,
            model=model,
            operation_id=request_id,
            operation_kind=kind,
        )
        try:
            payload = build_payload()
            if prepare is not None:
                prepare(record, payload)
            # Registration-pinned (anti-ABA): a stale model handle must fence,
            # never bind to a same-name successor registration.
            self.backend.enqueue_operation(model.name, request_id, seq_id, kind, payload, model.registration_id)
        except UserInputError as exc:
            # The client spent this ordinal: consume it as terminal
            # FAILED(user) so later operations never wait behind a gap.
            self._reject_into_ledger(record, model, seq_id, kind, request_dump, str(exc))
        except ValueError as exc:
            message = str(exc)
            if any(mark in message for mark in _LEDGER_CONFLICT_MARKS):
                raise ApiError(422, message) from exc
            if "not accepting operations" in message or "fenced" in message:
                record.resolve(wire.terminal_failure(message, "user"))
            else:
                self._reject_into_ledger(record, model, seq_id, kind, request_dump, message)
        self.futures.put(record)
        return wire.untyped_future(request_id, model.model_id)

    def _reject_into_ledger(
        self, record: FutureRecord, model: ModelRecord, seq_id: int, kind: str, request_dump: dict, error: str
    ) -> None:
        # The wire dump is the reject payload: deterministic across retries,
        # so a resend after a frontend restart matches the ledger fingerprint.
        try:
            self.backend.reject_operation(
                model.name, record.operation_id, seq_id, kind, {"wire": request_dump}, error, model.registration_id
            )
        except ValueError:
            record.resolve(wire.terminal_failure(error, "user"))

    # ---------------- checkpoints ----------------

    def weights_info(self, request: wire.WeightsInfoRequest) -> dict:
        checkpoint = self.checkpoints.get(request.tinker_path)
        if checkpoint is None:
            raise ApiError(
                404,
                f"unknown checkpoint '{request.tinker_path}'; v1 resolves paths minted during this service lifetime",
            )
        return {
            "base_model": checkpoint.base_model,
            "is_lora": True,
            "lora_rank": checkpoint.rank,
            "train_unembed": None,
            "train_mlp": None,
            "train_attn": None,
        }

    # ---------------- sampling ----------------

    def create_sampling_session(self, request: wire.CreateSamplingSessionRequest) -> dict:
        session = self.sessions.get(request.session_id)
        if session is None:
            raise ApiError(404, f"unknown session '{request.session_id}'")
        fingerprint = fingerprint_of(request.model_dump(mode="json"))
        sampling_session_id = f"samp-{session.short}-ss{request.sampling_session_seq_id}"
        try:
            existing = self.samplers.existing(sampling_session_id, fingerprint)
        except ConflictError as exc:
            raise ApiError(422, str(exc)) from exc
        if existing is not None:
            return {"type": "create_sampling_session", "sampling_session_id": sampling_session_id}
        if request.model_path is not None:
            raise ApiError(
                400,
                "sampling from saved checkpoints is not supported in v1 (latest-only serving); use "
                "save_weights_and_get_sampling_client on the training client, or a base_model session",
            )
        base_model = self._base_model()
        if request.base_model != base_model:
            raise ApiError(
                400, f"base_model '{request.base_model}' is not served; this deployment serves '{base_model}'"
            )
        self.samplers.add(
            SamplingSessionRecord(
                sampling_session_id=sampling_session_id,
                session_id=request.session_id,
                fingerprint=fingerprint,
                base_model=base_model,
            )
        )
        return {"type": "create_sampling_session", "sampling_session_id": sampling_session_id}

    def get_sampler(self, sampler_id: str) -> dict:
        sampler = self.samplers.get(sampler_id)
        if sampler is None:
            raise ApiError(404, f"unknown sampler '{sampler_id}'")
        return {"sampler_id": sampler.sampling_session_id, "base_model": sampler.base_model, "model_path": None}

    def sample(self, request: wire.SampleRequest) -> dict:
        if self._closing:
            raise ApiError(503, "the service is shutting down; no new samples are accepted")
        if request.sampling_session_id is None:
            raise ApiError(400, "asample requires sampling_session_id (create a sampling session first)")
        sampler = self.samplers.get(request.sampling_session_id)
        if sampler is None:
            raise ApiError(404, f"unknown sampler '{request.sampling_session_id}'")
        if request.seq_id is None or request.seq_id < 0:
            raise ApiError(400, "asample needs a seq_id >= 0")
        fingerprint = fingerprint_of(request.model_dump(mode="json"))
        request_id = f"{sampler.sampling_session_id}:s{request.seq_id}"
        if self._existing(request_id, fingerprint) is not None:
            return wire.untyped_future(request_id)
        if sampler.is_spent(request.seq_id):
            # The replay bytes AND the fingerprint tombstone are gone (bounded
            # retention rolled over), but the per-session spent-sequence fence
            # still knows this identity executed: answer a typed terminal
            # failure instead of silently re-running the generation.
            record = self.futures.put(FutureRecord(request_id=request_id, kind="sample", fingerprint=fingerprint))
            record.resolve(
                wire.terminal_failure(
                    f"sample seq {request.seq_id} of '{sampler.sampling_session_id}' was already executed "
                    "and its result expired from the replay window; it cannot be re-run",
                    "user",
                )
            )
            return wire.untyped_future(request_id)

        record = FutureRecord(request_id=request_id, kind="sample", fingerprint=fingerprint)
        try:
            if request.topk_prompt_logprobs:
                raise UserInputError("topk_prompt_logprobs is not supported in v1")
            if request.num_samples < 1:
                raise UserInputError("num_samples must be >= 1")
            prompt_tokens = translation._input_tokens("prompt", request.prompt)
            sglang_params = translation.sampling_params_to_sglang(request.sampling_params)
            seed = request.sampling_params.seed
            if seed is not None and seed + request.num_samples - 1 >= 2**63:
                raise UserInputError("sampling_params.seed + num_samples must fit in a signed 64-bit integer")
        except UserInputError as exc:
            # Invalid payloads still consume the seq as a typed terminal (the
            # http_server contract) — but never a permit: nothing will run.
            sampler.mark_spent(request.seq_id)
            self.futures.put(record)
            record.resolve(wire.terminal_failure(str(exc), "user"))
            return wire.untyped_future(request_id)

        admission = self.sampling_admission
        if request.num_samples > admission.capacity:
            # Would 429 forever — fail typed and non-retryable, without
            # consuming the seq, so the client can split into waves.
            raise ApiError(
                400,
                f"num_samples={request.num_samples} exceeds this deployment's sampling capacity of "
                f"{admission.capacity} concurrent sub-generations; split the request into smaller waves",
            )
        # Context preflight (code-0815 §6.2): a prompt that leaves no decode
        # budget must fail HERE, typed and non-retryable — the engine itself
        # silently truncates max_new_tokens to whatever fits (near zero for
        # an oversized accumulated context) and returns garbage. Like the
        # num_samples cap above: a deterministic 400 before the seq identity
        # is consumed, so nothing executes and nothing gaps.
        limit = self._context_limit
        if limit is None:
            self._ensure_context_limit_discovery()
        else:
            max_new_tokens = sglang_params["max_new_tokens"]
            if len(prompt_tokens) + max_new_tokens > limit:
                raise ApiError(
                    400,
                    f"prompt ({len(prompt_tokens)} tokens) + max_tokens ({max_new_tokens}) exceeds this "
                    f"deployment's engine context limit of {limit} tokens ({self._context_limit_source}); "
                    "shorten the prompt or lower max_tokens — the engine would silently truncate the "
                    "decode budget instead of honoring the request",
                )
        if not admission.try_acquire(request.num_samples):
            # BEFORE mark_spent/FutureRecord: the identity stays unconsumed,
            # so the SDK's backoff retry of the SAME seq id is safe. The HTTP
            # layer maps this to 429 + Retry-After.
            raise OperationBackpressure(
                f"sampling capacity reached ({admission.in_use}/{admission.capacity} sub-generations "
                "active); retry the identical request"
            )
        # No await from try_acquire to create_task: admission, identity
        # consumption, and FutureRecord creation are one atomic submission
        # step (two identical racing requests cannot both execute).
        sampler.mark_spent(request.seq_id)
        self.futures.put(record)
        task = asyncio.get_running_loop().create_task(
            self._run_sample(
                record,
                sampler,
                prompt_tokens,
                sglang_params,
                request.num_samples,
                request.sampling_params.seed,
                prompt_logprobs=bool(request.prompt_logprobs),
            )
        )
        self._sample_tasks.add(task)
        self._sample_task_by_request[request_id] = task
        task.add_done_callback(
            lambda done: self._terminalize_prestart_cancelled_sample(
                done,
                record,
                request.num_samples,
                len(prompt_tokens),
                sglang_params.get("max_new_tokens"),
            )
        )
        task.add_done_callback(self._sample_tasks.discard)
        task.add_done_callback(lambda _task, rid=request_id: self._sample_task_by_request.pop(rid, None))
        # Release via done-callback, not inside the coroutine: a task
        # cancelled before its first step never enters the coroutine body, so
        # a `finally` there could leak the permit on shutdown.
        task.add_done_callback(lambda _task, weight=request.num_samples: admission.release(weight))
        return wire.untyped_future(request_id)

    # ---------------- engine context discovery ----------------

    _CONTEXT_DISCOVERY_MAX_ATTEMPTS = 3

    def _ensure_context_limit_discovery(self) -> None:
        """Single-flight, non-blocking: sample submission stays synchronous
        (admission atomicity), so discovery runs as a background task kicked
        off by the first sample. Until it lands the preflight admits
        everything (a permissive window, never a false reject)."""
        if (
            self._context_limit is not None
            or self._closing
            or self._context_discovery_task is not None
            or self._context_discovery_attempts >= self._CONTEXT_DISCOVERY_MAX_ATTEMPTS
        ):
            return
        server_info = getattr(self.sampling_transport, "server_info", None)
        if server_info is None:
            self._context_discovery_attempts = self._CONTEXT_DISCOVERY_MAX_ATTEMPTS
            logger.warning(
                "[tinker] sampling context preflight disabled: the sampling transport exposes no "
                "server_info; pass --tinker-sampling-max-context to enforce a limit"
            )
            return
        self._context_discovery_task = asyncio.get_running_loop().create_task(
            self._discover_context_limit(server_info)
        )

    async def _discover_context_limit(self, server_info: Callable) -> None:
        self._context_discovery_attempts += 1
        attempt = f"attempt {self._context_discovery_attempts}/{self._CONTEXT_DISCOVERY_MAX_ATTEMPTS}"
        try:
            info = await server_info()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — discovery must never take sampling down
            if self._context_discovery_attempts >= self._CONTEXT_DISCOVERY_MAX_ATTEMPTS:
                logger.warning(
                    f"[tinker] sampling context preflight disabled: engine context discovery failed "
                    f"({attempt}: {type(exc).__name__}: {exc}); pass --tinker-sampling-max-context "
                    "to enforce a limit"
                )
            else:
                logger.info(f"[tinker] engine context discovery failed ({attempt}), will retry: {exc}")
            return
        finally:
            # Cleared AFTER the outcome is recorded: the next sample may
            # re-trigger discovery only while attempts remain.
            self._context_discovery_task = None
        limit = _context_limit_from_server_info(info)
        if limit is None:
            self._context_discovery_attempts = self._CONTEXT_DISCOVERY_MAX_ATTEMPTS
            logger.warning(
                "[tinker] sampling context preflight disabled: /get_server_info carried neither "
                "context_length nor max_req_input_len; pass --tinker-sampling-max-context to enforce a limit"
            )
            return
        self._context_limit = limit
        self._context_limit_source = "discovered from the engine"
        logger.info(f"[tinker] sampling context preflight active: engine context limit {limit} tokens (discovered)")

    def _terminalize_prestart_cancelled_sample(
        self,
        task: asyncio.Task,
        record: FutureRecord,
        num_samples: int,
        prompt_tokens: int,
        max_new_tokens: int | None,
    ) -> None:
        """Resolve a task cancelled before its coroutine body ever ran."""
        if not task.cancelled() or record.terminal is not None:
            return
        record.failure_class = "Cancelled"
        record.resolve(
            wire.terminal_failure(record.cancel_reason or "sampling cancelled: the service is shutting down", "server")
        )
        self._account_sample_terminal(record, num_samples, prompt_tokens, max_new_tokens)

    async def _run_sample(
        self,
        record: FutureRecord,
        sampler: SamplingSessionRecord,
        tokens: list[int],
        params: dict,
        num_samples: int,
        seed: int | None = None,
        prompt_logprobs: bool = False,
    ) -> None:
        try:
            await self._execute_sample(record, sampler, tokens, params, num_samples, seed, prompt_logprobs)
        except asyncio.CancelledError:
            # Reaper cancellation carries its reason on the record; anything
            # else is the shutdown barrier. Either way the future resolves so
            # a client polling it sees a typed terminal, never an identity
            # that silently stops progressing.
            record.failure_class = "Cancelled"
            record.resolve(
                wire.terminal_failure(
                    record.cancel_reason or "sampling cancelled: the service is shutting down", "server"
                )
            )
            raise
        except Exception as exc:  # noqa: BLE001 — every failure must resolve the future
            # Always name the exception class: str(httpx.PoolTimeout()) is
            # empty, and a bare "sampling failed: " is undiagnosable.
            record.failure_class = type(exc).__name__
            record.resolve(wire.terminal_failure(f"sampling failed ({type(exc).__name__}): {exc}", "server"))
        finally:
            self._account_sample_terminal(record, num_samples, len(tokens), params.get("max_new_tokens"))

    async def _execute_sample(
        self,
        record: FutureRecord,
        sampler: SamplingSessionRecord,
        tokens: list[int],
        params: dict,
        num_samples: int,
        seed: int | None = None,
        prompt_logprobs: bool = False,
    ) -> None:
        payload: dict = {"input_ids": tokens, "sampling_params": params, "return_logprob": True}
        if prompt_logprobs:
            # sglang natively scores the prompt: input_token_logprobs from position 0.
            payload["logprob_start_len"] = 0
        if sampler.name is not None:
            live = self.backend.registration_view(sampler.name)
            if live is None or live["registration_id"] != sampler.registration_id:
                record.resolve(
                    wire.terminal_failure("sampler weights are no longer live (registration retired)", "user")
                )
                return
            if live["serving_version"] != sampler.serving_version:
                record.resolve(
                    wire.terminal_failure(
                        "stale ephemeral sampler: the model was republished and this backend serves the "
                        "latest weights only — create a new sampling client after each publish",
                        "user",
                    )
                )
                return
            payload["lora_path"] = sampler.serving_name
            payload["extra_key"] = cache_extra_key(sampler.name, sampler.registration_id, sampler.serving_version)

        def per_sample_payload(index: int) -> dict:
            one = dict(payload)
            if seed is not None:
                # Deterministic per request, still diverse across samples.
                one["sampling_params"] = {**params, "sampling_seed": seed + index}
            if sampler.name is not None:
                one["rid"] = make_rid(sampler.name, sampler.registration_id)
            return one

        # Not a bare gather: the first exception must not leave siblings
        # running untracked — cancel them and AWAIT their cancellation
        # before this future turns terminal, so no generation outlives
        # its request's resolution.
        generation_tasks = [
            asyncio.get_running_loop().create_task(self.sampling_transport.generate(per_sample_payload(index)))
            for index in range(num_samples)
        ]
        for generation_task in generation_tasks:
            generation_task.add_done_callback(lambda task, r=record: _note_first_result(task, r))
        try:
            generations = await asyncio.gather(*generation_tasks)
        except BaseException:
            for task in generation_tasks:
                task.cancel()
            await asyncio.gather(*generation_tasks, return_exceptions=True)
            raise
        if sampler.name is not None and not self._sampler_still_live(sampler):
            # Re-checked AFTER generation: a republish that landed while
            # the request was in flight swapped the engine-side weights
            # under the same serving name (latest-only serving), so the
            # output cannot be attributed to the pinned version. Fail loud
            # rather than return cross-version samples. (A publish
            # committing between this check and delivery remains possible
            # — the serving identity is versioned, not leased; see README.)
            record.resolve(
                wire.terminal_failure(
                    "the model was republished while this sample was in flight; create a new sampling "
                    "client after each publish and resample",
                    "user",
                )
            )
            return
        sequences = [translation.generation_to_sequence(generation) for generation in generations]
        # The prompt is shared across the fan-out, so any generation's scores serve.
        scored = translation.prompt_logprobs_from_generation(generations[0], len(tokens)) if prompt_logprobs else None
        record.resolve(translation.sequences_to_sample_response(sequences, scored))

    def _account_sample_terminal(
        self, record: FutureRecord, num_samples: int, prompt_tokens: int, max_new_tokens: int | None
    ) -> None:
        """Single terminal choke point for every task-executed sample: the
        §6.1 counters plus one per-request line carrying the latencies. Per
        request at DEBUG (high-volume), failures at WARNING with their class."""
        body = record.terminal or {}
        stats = self.sampling_stats
        admission = self.sampling_admission
        terminal_at = record.resolved_at or time.time()
        total_s = terminal_at - record.created_at
        first_result_s = (record.first_result_at - record.created_at) if record.first_result_at is not None else None
        first_result = f"{first_result_s:.3f}" if first_result_s is not None else "n/a"
        detail = (
            f"request='{record.request_id}' num_samples={num_samples} prompt_tokens={prompt_tokens} "
            f"max_tokens={max_new_tokens} queue_to_first_result_s={first_result} total_s={total_s:.3f} "
            f"active={admission.in_use}/{admission.capacity} peak={admission.peak_in_use} "
            f"admitted={admission.admitted} rejected_429={admission.rejected}"
        )
        stats.record_latency(first_result_s, total_s)
        if "error" in body:
            failure_class = record.failure_class or body.get("category") or "unknown"
            stats.record_failure(failure_class)
            logger.warning(
                f"[tinker] sample terminal failure class={failure_class} category={body.get('category')} "
                f"{detail} error={body.get('error')!r}"
            )
        else:
            stats.completed += 1
            logger.debug(f"[tinker] sample terminal ok {detail}")

    def _sampler_still_live(self, sampler: SamplingSessionRecord) -> bool:
        live = self.backend.registration_view(sampler.name)
        return (
            live is not None
            and live["registration_id"] == sampler.registration_id
            and live["serving_version"] == sampler.serving_version
        )

    # ---------------- future retrieval ----------------

    async def retrieve_future(self, request: wire.FutureRetrieveRequest) -> dict:
        """Long-poll: resolve inside the window when possible, else try_again."""
        deadline = time.monotonic() + self.poll_window_s
        while True:
            record = self.futures.get(request.request_id)
            if record is None:
                if self.futures.expired_fingerprint(request.request_id) is not None:
                    raise ApiError(
                        410,
                        f"request '{request.request_id}' was already delivered and its replay window expired",
                    )
                if self.futures.reaped_fingerprint(request.request_id) is not None:
                    raise ApiError(
                        410,
                        f"request '{request.request_id}' completed but was never retrieved within its "
                        "retention TTL and was reaped",
                    )
                raise ApiError(
                    410, f"unknown request '{request.request_id}' (expired or from a previous service lifetime)"
                )
            # Liveness for the orphan reaper: an actively polled future is
            # never an orphan, whatever its age.
            record.last_polled_at = time.time()
            if record.terminal is None:
                self._poll(record)
            if record.terminal is not None:
                body = record.terminal
                self.futures.mark_delivered(record)
                return body
            if time.monotonic() >= deadline:
                return wire.try_again(self._queue_state(record))
            await asyncio.sleep(self.poll_interval_s)

    def _queue_state(self, record: FutureRecord) -> str:
        if record.kind == "create_model" and record.model is not None:
            live = self.backend.registration_view(record.model.name)
            if live is not None and not live["bound"]:
                return "paused_capacity"
        return "active"

    def _poll(self, record: FutureRecord) -> None:
        if record.kind == "operation":
            self._poll_operation(record)
        elif record.kind == "create_model":
            self._poll_create_model(record)
        elif record.kind == "unload_model":
            self._poll_unload_model(record)
        # "sample" resolves from its own task.

    def _poll_operation(self, record: FutureRecord) -> None:
        view = self.backend.operation_view(record.operation_id)
        if view is None:
            record.resolve(wire.terminal_failure("operation record lost before retrieval", "server"))
            return
        state = view["state"]
        if state in ("QUEUED", "CLAIMED"):
            return
        if state == "SUCCEEDED":
            record.resolve(self._success_body(record, view.get("result") or {}))
        else:  # FAILED | CANCELLED
            error = view.get("error") or "operation failed"
            if record.backend_target and record.tinker_path:
                # Clients know the tinker:// URI, not the trainer's filesystem.
                error = error.replace(record.backend_target["path"], record.tinker_path)
            record.resolve(wire.terminal_failure(error, view.get("error_category") or "server"))
        # Ack only after the terminal body is stored: a lost response replays
        # from the future store, never from a record the ack released.
        self.backend.ack_operation(record.operation_id)

    def _success_body(self, record: FutureRecord, result: dict) -> dict:
        kind, model = record.operation_kind, record.model
        if kind in ("forward_backward", "forward"):
            return translation.fb_result_to_response(result, record.forward_payload)
        if kind == "optim_step":
            return translation.optim_result_to_response(result)
        if kind == "save_state":
            backend_path = str(result.get("path"))
            tag = backend_path.rstrip("/").rsplit("/", 1)[-1]
            tinker_path = f"tinker://{model.name}.{model.rid8}/weights/{tag}"
            self.checkpoints.add(
                CheckpointRecord(
                    tinker_path=tinker_path,
                    backend_path=backend_path,
                    name=model.name,
                    registration_id=model.registration_id,
                    base_model=model.base_model,
                    rank=model.rank,
                    step=int(result.get("step") or 0),
                )
            )
            return translation.save_weights_result_to_response(tinker_path)
        if kind == "load_state":
            return translation.load_weights_result_to_response(record.tinker_path, model.model_id)
        if kind == "save_weights_for_sampler":
            if self.sessions.get(model.session_id) is None:
                return wire.terminal_failure(
                    "the parent session expired before sampler publication completed; create a new session", "user"
                )
            existing = self.samplers.get(record.sampling_session_id)
            if existing is not None and existing.fingerprint != record.fingerprint:
                # Never overwrite a live sampler identity: a base sampler (or
                # another publish) already owns this namespace, and silently
                # rebinding it would swap the weights under an existing
                # client. The weights are live (the publish itself landed);
                # only the sampler minting fails, typed.
                return wire.terminal_failure(
                    f"sampling session '{record.sampling_session_id}' already exists; publish with a fresh "
                    "sampling_session_seq_id to mint a new sampler",
                    "user",
                )
            self.samplers.add(
                SamplingSessionRecord(
                    sampling_session_id=record.sampling_session_id,
                    session_id=model.session_id,
                    fingerprint=record.fingerprint,
                    base_model=model.base_model,
                    name=model.name,
                    registration_id=model.registration_id,
                    serving_name=result.get("serving_name") or serving_lora_name(model.name, model.registration_id),
                    serving_version=result.get("serving_version"),
                )
            )
            return translation.sampler_publish_result_to_response(record.sampling_session_id)
        return wire.terminal_failure(f"no translator for operation kind '{kind}'", "server")

    def _poll_create_model(self, record: FutureRecord) -> None:
        model = record.model
        live = self.backend.registration_view(model.name)
        if live is None or live["registration_id"] != model.registration_id:
            record.resolve(wire.terminal_failure("registration retired before the model became ready", "user"))
            return
        if live["state"] == "READY":
            record.resolve({"type": "create_model", "model_id": model.model_id})
        elif live["state"] != "PENDING":
            record.resolve(wire.terminal_failure(f"registration is {live['state']}; model creation failed", "user"))

    def _poll_unload_model(self, record: FutureRecord) -> None:
        model = record.model
        live = self.backend.registration_view(model.name)
        if live is None or live["registration_id"] != model.registration_id:
            record.resolve({"type": "unload_model", "model_id": model.model_id})
