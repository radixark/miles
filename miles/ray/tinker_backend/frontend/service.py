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
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any


from miles.ray.tinker_backend.config import AdapterRunConfig
from miles.ray.tinker_backend.frontend import translation, wire
from miles.ray.tinker_backend.frontend.sampling import SamplingTransport, SGLangRouterSamplingTransport
from miles.ray.tinker_backend.frontend.state import (
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
from miles.ray.tinker_backend.frontend.translation import UserInputError
from miles.utils.tinker_backend import cache_extra_key, make_rid, serving_lora_name

logger = logging.getLogger(__name__)

_LEDGER_CONFLICT_MARKS = ("different content", "already taken")


class ApiError(Exception):
    """Maps to an HTTP error response (submit-time failures the SDK should
    see as a status code, not a terminal future)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class TinkerFrontend:
    """One instance per controller; single event loop, no cross-await state
    mutation inside a submit or resolve step."""

    def __init__(
        self,
        backend: Any,
        poll_window_s: float = 15.0,
        poll_interval_s: float = 0.1,
        sampling_transport: SamplingTransport | None = None,
    ) -> None:
        self.backend = backend
        self.poll_window_s = poll_window_s
        self.poll_interval_s = poll_interval_s
        # Injected sampling hop (frontend -> router); the default preserves
        # the direct-router transport this frontend always used.
        self.sampling_transport = (
            sampling_transport
            if sampling_transport is not None
            else SGLangRouterSamplingTransport(
                backend.sampling_endpoint(),
                max_connections=int(getattr(backend.args, "router_queue_size", 100)),
                pool_timeout_s=float(getattr(backend.args, "router_queue_timeout_secs", 600.0)),
            )
        )
        self.sessions = SessionStore()
        self.models = ModelStore()
        self.futures = FutureStore()
        self.checkpoints = CheckpointCatalog()
        self.samplers = SamplingSessionStore()
        self._sample_tasks: set[asyncio.Task] = set()
        self._closing = False

    async def close(self) -> None:
        """Idempotent shutdown barrier: gate new samples, cancel AND await
        every in-flight sample task (so the transport observes cancellation
        before it is closed under it), then close the transport."""
        self._closing = True
        tasks = list(self._sample_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            # The done-callbacks discard too, but only on a later loop tick;
            # close() must return with the set verifiably drained.
            self._sample_tasks.difference_update(tasks)
        await self.sampling_transport.close()

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
        model = {"model_name": info.get("base_model"), "max_context_length": None}
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
        sampler.mark_spent(request.seq_id)

        record = self.futures.put(FutureRecord(request_id=request_id, kind="sample", fingerprint=fingerprint))
        try:
            if request.prompt_logprobs:
                raise UserInputError("prompt_logprobs is not supported in v1")
            if request.topk_prompt_logprobs:
                raise UserInputError("topk_prompt_logprobs is not supported in v1")
            if request.num_samples < 1:
                raise UserInputError("num_samples must be >= 1")
            prompt_tokens = translation._input_tokens("prompt", request.prompt)
            sglang_params = translation.sampling_params_to_sglang(request.sampling_params)
        except UserInputError as exc:
            record.resolve(wire.terminal_failure(str(exc), "user"))
            return wire.untyped_future(request_id)

        task = asyncio.get_running_loop().create_task(
            self._run_sample(
                record, sampler, prompt_tokens, sglang_params, request.num_samples, request.sampling_params.seed
            )
        )
        self._sample_tasks.add(task)
        task.add_done_callback(self._sample_tasks.discard)
        return wire.untyped_future(request_id)

    async def _run_sample(
        self,
        record: FutureRecord,
        sampler: SamplingSessionRecord,
        tokens: list[int],
        params: dict,
        num_samples: int,
        seed: int | None = None,
    ) -> None:
        try:
            payload: dict = {"input_ids": tokens, "sampling_params": params, "return_logprob": True}
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
            record.resolve(translation.sequences_to_sample_response(sequences))
        except asyncio.CancelledError:
            # Shutdown cancellation: resolve so a client polling the future
            # sees a typed terminal instead of an identity that never lands.
            record.resolve(wire.terminal_failure("sampling cancelled: the service is shutting down", "server"))
            raise
        except Exception as exc:  # noqa: BLE001 — every failure must resolve the future
            record.resolve(wire.terminal_failure(f"sampling failed: {exc}", "server"))

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
                raise ApiError(
                    410, f"unknown request '{request.request_id}' (expired or from a previous service lifetime)"
                )
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
