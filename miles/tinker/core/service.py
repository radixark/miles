"""Sessions, models, futures, and ordered trainer dispatch.

The trainer lock serializes trainer calls across dispatch, model creation, and lease expiry."""

import asyncio
import logging
import os
import secrets
import time
import uuid
from datetime import UTC, datetime
from typing import NamedTuple

from miles.tinker.core.checkpoint_expiry import CheckpointExpiry
from miles.tinker.core.future import RequestFuture, RequestFutureStore
from miles.tinker.core.input_validation import (
    MIN_TTL_S,
    validate_batch_payload,
    validate_checkpoint_segment,
    validate_model_config,
    validate_sample_payload,
    validate_seq_id,
)
from miles.tinker.core.model_queue import ModelRequestQueue
from miles.tinker.core.scheduler import BarrierUnit, BatchUnit, RequestScheduler
from miles.tinker.core.types import (
    Command,
    CommandOp,
    GatewayConfig,
    ModelRecord,
    NotFoundError,
    OwnershipError,
    SamplingSessionRecord,
    SessionRecord,
    UserInputError,
)
from miles.tinker.core.utils import (
    build_parametrization_fields,
    parse_tinker_path,
    read_checkpoint_metadata,
    resolve_checkpoint_dir,
    resolve_sampler_checkpoint,
    resolve_sampler_export,
    tenant_digest,
)

logger = logging.getLogger(__name__)

# how long the Tinker SDK keeps retrying a finish (InternalClientHolder.execute_with_retries)
_SDK_FINISH_RETRY_S = 300.0


class _EndedSession(NamedTuple):
    tenant: str
    lease_expired: bool
    ended_at: float


class _Download(NamedTuple):
    tenant: str
    tinker_path: str
    # a re-save under the same name after a delete is another checkpoint, which this URL must not serve
    commit_id: str
    expires_at: float


# as official Tinker: a download URL lives an hour, and never past its checkpoint's expiry
_DOWNLOAD_URL_TTL_S = 3600
# a TTL set after a URL was signed can shorten the checkpoint's life to no less than this, so no URL outlives it
assert _DOWNLOAD_URL_TTL_S <= MIN_TTL_S
_EXPIRY_SWEEP_S = 60


class TinkerService:
    def __init__(self, backend, config: GatewayConfig) -> None:
        self.backend = backend
        self.config = config
        self.futures = RequestFutureStore()
        self.scheduler = RequestScheduler(config.batch_token_budget)
        self.models: dict[str, ModelRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.sampling_sessions: dict[str, SamplingSessionRecord] = {}
        self.free_slots = set(range(config.n_slots))
        self._wake = asyncio.Event()
        self._trainer_lock = asyncio.Lock()
        self._sample_tasks: dict[str, tuple] = {}  # request_id -> (task, session_id)
        self._create_tasks: set = set()
        self._cleanup_tasks: set = set()
        # two cleanups would race on the same .deleting dirs
        self._cleanup_lock = asyncio.Lock()
        self._arrival_counter = 0
        self._batch_counter = 0
        self._background_error: BaseException | None = None
        # why each model closed, so later requests get the reason instead of "unknown model"
        self._close_reasons: dict[str, str] = {}
        # ended sessions, kept a lease long and at least the SDK's retry window so a retried finish is answered
        self._ended_sessions: dict[str, _EndedSession] = {}
        self.expiry = CheckpointExpiry(config.checkpoint_root)
        # expiry changes and deletes; never held across a training call
        self._expiry_lock = asyncio.Lock()
        # issued download tokens; unguessable, held only in memory, so a restart invalidates every URL
        self._downloads: dict[str, _Download] = {}
        # the token handed out per checkpoint commit, so repeated requests reuse it; a replaced one lives to its expiry
        self._download_tokens: dict[tuple[str, str, str], str] = {}

    async def run(self) -> None:
        await self._expire_checkpoints()
        sweep_task = asyncio.create_task(self._run_lease_sweeper())
        sweep_task.add_done_callback(self._observe_background_task)
        expiry_task = asyncio.create_task(self._run_expiry_sweeper())
        expiry_task.add_done_callback(self._observe_background_task)
        try:
            while True:
                if self._background_error is not None:
                    raise self._background_error
                # unit selection shares the critical section with execution, so
                # lease expiry cannot reclaim a model queue between the two
                async with self._trainer_lock:
                    rejections = self.scheduler.ready_rejections()
                    if rejections:
                        for model_queue, pending in rejections:
                            await self._finish_request(
                                model_queue,
                                pending,
                                {"error": pending.command.validation_error, "error_category": "user"},
                            )
                        continue
                    unit = self.scheduler.schedule_next()
                    if unit is not None:
                        if isinstance(unit, BatchUnit):
                            await self._run_batch(unit)
                        else:
                            await self._run_barrier(unit)
                        if await self.backend.trainer_dead():
                            raise RuntimeError("the trainer workers died; exiting so clients get refused connections")
                        continue
                await self._wake.wait()
                self._wake.clear()
        finally:
            tasks = [sweep_task, expiry_task, *self._create_tasks, *(task for task, _ in self._sample_tasks.values())]
            for task in tasks:
                task.cancel()
            # Cleanup failures must not replace the error that stopped the dispatcher.
            # awaited, not cancelled: a cancelled to_thread rmtree would outlive the root lock
            await asyncio.gather(*tasks, *self._cleanup_tasks, return_exceptions=True)

    def _observe_background_task(self, task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            if self._background_error is None:
                self._background_error = error
            self._wake.set()

    def create_session(self, tenant: str) -> str:
        session_id = f"session-{uuid.uuid4().hex}"
        self.sessions[session_id] = SessionRecord(tenant=tenant, last_heartbeat=time.monotonic())
        return session_id

    def _session_for(self, tenant: str, session_id: str) -> SessionRecord:
        session = self.sessions.get(session_id)
        if session is None or session.tenant != tenant:
            raise UserInputError(f"unknown session {session_id!r}; create a session first")
        return session

    def heartbeat(self, tenant: str, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None or session.tenant != tenant:
            return False
        session.last_heartbeat = time.monotonic()
        return True

    def create_model(self, tenant: str, payload: dict) -> tuple[str, str]:
        """Two-phase like every command: allocate now, initialize the slot behind the future."""
        session = self._session_for(tenant, payload["session_id"])
        model_seq_id = validate_seq_id(payload["model_seq_id"], "model_seq_id", minimum=0)
        if (previous := session.models_by_seq.get(model_seq_id)) is not None:
            request_id, model_id = previous
            request_id = self.futures.request_id_for_retry(request_id, model_id, tenant)
            session.models_by_seq[model_seq_id] = (request_id, model_id)
            return request_id, model_id
        base_model = payload["base_model"]
        if base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        lora_config = payload.get("lora_config") or {}
        validate_model_config(lora_config, self.config)
        rank = lora_config.get("rank", 32)
        alpha = self.config.lora_alpha if self.config.lora_alpha is not None else float(2 * rank)
        if not self.free_slots:
            raise UserInputError(f"no free adapter slots (capacity {self.config.n_slots})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)

        model_id = f"model-{uuid.uuid4().hex[:12]}"
        future = self.futures.create(model_id, tenant)
        record = ModelRecord(
            model_id=model_id,
            session_id=payload["session_id"],
            tenant=tenant,
            slot=slot,
            base_model=base_model,
            lora_rank=rank,
            lora_alpha=alpha,
            create_request_id=future.request_id,
        )
        self.models[model_id] = record
        self.scheduler.add_model_queue(ModelRequestQueue(model_id, tenant, slot))
        task = asyncio.create_task(self._run_create_model(record))
        self._create_tasks.add(task)
        task.add_done_callback(self._create_tasks.discard)
        task.add_done_callback(self._observe_background_task)
        session.models_by_seq[model_seq_id] = (future.request_id, model_id)
        return future.request_id, model_id

    async def _run_create_model(self, record: ModelRecord) -> None:
        async with self._trainer_lock:
            if self.models.get(record.model_id) is not record:
                return
            failure = await self.backend.load_slot(record.slot, record.lora_rank, record.lora_alpha)
            record.slot_initialized = True
            if failure is not None:
                await self._close_model(record.model_id, failure["error"], "server")
                return
            self.futures.resolve(record.create_request_id, {"op": "create_model", "model_id": record.model_id})

    def get_model(self, tenant: str, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            reason = self._close_reasons.get(model_id)
            if reason is not None:
                raise UserInputError(f"model {model_id!r} was unloaded: {reason}")
            raise UserInputError(f"unknown model {model_id!r}")
        if record.tenant != tenant:
            raise OwnershipError(f"model {model_id} does not belong to this tenant")
        return record

    async def _close_model(self, model_id: str, error: str, category: str) -> None:
        """Free a model's slot and fail its pending requests; requires the trainer lock. Idempotent."""
        record = self.models.pop(model_id, None)
        if record is None:
            return
        self._close_reasons[model_id] = error
        while len(self._close_reasons) > 4 * self.config.n_slots:
            self._close_reasons.pop(next(iter(self._close_reasons)))
        self.scheduler.remove_model_queue(model_id)
        for request_id in [record.create_request_id, *record.request_id_by_seq.values()]:
            if self.futures.get(request_id, record.tenant) is not None:
                self.futures.fail(request_id, error, category)
        if await self.backend.trainer_dead():
            return
        if record.slot_initialized:
            failure = await self.backend.unload_slot(record.slot)
            if failure is not None:
                logger.error("slot %s remains unavailable after unload failed: %s", record.slot, failure["error"])
                return
        self.free_slots.add(record.slot)

    def submit(self, tenant: str, op: str, payload: dict) -> str:
        """Admit a decoded command; content errors settle its future as a user failure."""
        try:
            op = CommandOp(op)
        except ValueError:
            raise UserInputError(f"unknown command op {op!r}") from None
        model_id = payload["model_id"]
        record = self.get_model(tenant, model_id)
        seq_id = validate_seq_id(payload["seq_id"], "seq_id")
        model_queue = self.scheduler.model_queue(model_id)

        # retries must not accumulate gradients twice
        if seq_id in record.request_id_by_seq:
            request_id = self.futures.request_id_for_retry(record.request_id_by_seq[seq_id], model_id, tenant)
            record.request_id_by_seq[seq_id] = request_id
            return request_id

        future = self.futures.create(model_id, tenant)
        record.request_id_by_seq[seq_id] = future.request_id
        self._arrival_counter += 1
        validation_error = payload.get("validation_error")
        if validation_error is None:
            try:
                validate_batch_payload(op, payload, self.config)
            except UserInputError as error:
                validation_error = str(error)
        model_queue.submit(
            Command(
                model_id=model_id,
                seq_id=seq_id,
                op=op,
                payload={key: value for key, value in payload.items() if key != "validation_error"},
                request_id=future.request_id,
                arrival=self._arrival_counter,
                validation_error=validation_error,
            )
        )
        self._wake.set()
        return future.request_id

    def retrieve_future(self, tenant: str, request_id: str) -> RequestFuture | None:
        return self.futures.get(request_id, tenant)

    async def _run_batch(self, batch: BatchUnit) -> None:
        # slot-contiguous order; outputs come back aligned to it
        refs = sorted(batch.datums, key=lambda ref: ref.model_queue.slot)
        slot_datums = [(ref.model_queue.slot, ref.datum) for ref in refs]
        self._batch_counter += 1
        forward = (
            self.backend.forward_backward if batch.op == CommandOp.FORWARD_BACKWARD else self.backend.forward_only
        )
        try:
            outputs = await forward(self._batch_counter, slot_datums, batch.loss_fn, batch.loss_fn_config)
        except UserInputError as error:
            await self._fail_batch(batch, str(error), "user")
            return
        if isinstance(outputs, dict) and "error" in outputs:
            await self._fail_batch(batch, outputs["error"], "server")
            return

        assert len(outputs) == len(refs), f"unit returned {len(outputs)} outputs for {len(refs)} datums"
        for ref, output in zip(refs, outputs, strict=True):
            request = ref.request
            if request.record_output(ref.local_index, output):
                await self._finish_request(
                    ref.model_queue, request, {"op": request.command.op, "outputs": request.outputs}
                )

    async def _fail_batch(self, batch: BatchUnit, error: str, category: str) -> None:
        requests = {ref.request.command.request_id: (ref.model_queue, ref.request) for ref in batch.datums}
        for model_queue, pending in requests.values():
            await self._finish_request(model_queue, pending, {"error": error, "error_category": category})

    async def _run_barrier(self, barrier: BarrierUnit) -> None:
        try:
            outcomes = await self._dispatch_barrier_op(barrier)
        except (UserInputError, OwnershipError) as error:
            outcomes = [{"error": str(error), "error_category": "user"} for _ in barrier.entries]
        for (model_queue, pending), outcome in zip(barrier.entries, outcomes, strict=True):
            await self._finish_request(model_queue, pending, outcome)

    async def _dispatch_barrier_op(self, barrier: BarrierUnit) -> list[dict]:
        """Return one result or error per entry; only the caller settles futures and retires models."""
        if barrier.op == CommandOp.OPTIM_STEP:
            return await self._step_optimizers(barrier.entries)
        ((model_queue, pending),) = barrier.entries  # every other barrier is single-entry
        record = self.models[model_queue.model_id]
        payload = pending.command.payload
        if barrier.op == CommandOp.SAVE_STATE:
            return [await self._save_state(record, payload)]
        if barrier.op == CommandOp.LOAD_STATE:
            return [await self._load_state(record, payload)]
        if barrier.op == CommandOp.SAVE_WEIGHTS_FOR_SAMPLER:
            return [await self._save_weights_for_sampler(record, payload)]
        if barrier.op == CommandOp.SAVE_WEIGHTS_EXTERNAL:
            return [await self._save_weights_external(record, payload)]
        raise UserInputError(f"unknown barrier op {barrier.op!r}")

    async def _step_optimizers(self, entries: list) -> list[dict]:
        adam_params_by_slot = {
            model_queue.slot: pending.command.payload["adam_params"] for model_queue, pending in entries
        }
        slot_outcomes = await self.backend.optim_step(adam_params_by_slot)
        outcomes = []
        for model_queue, _ in entries:
            outcome = slot_outcomes[model_queue.slot]
            if "error" in outcome:
                outcomes.append(outcome)
            else:
                outcomes.append({"op": "optim_step", "metrics": {key: float(value) for key, value in outcome.items()}})
        return outcomes

    async def _finish_request(self, model_queue, pending, outcome: dict) -> None:
        if "error" in outcome:
            category = outcome.get("error_category", "server")
            if pending.command.op.requires_model_close_on_failure(category):
                await self._close_model(
                    model_queue.model_id,
                    f"model training failed ({outcome['error']}); create a new model and restore from a checkpoint",
                    category,
                )
                return
            self.futures.fail(pending.command.request_id, outcome["error"], category)
        else:
            self.futures.resolve(pending.command.request_id, outcome)
        model_queue.finish(pending)

    async def _save_state(self, record: ModelRecord, payload: dict) -> dict:
        """Save parameters and optimizer state; call after optim_step to persist accumulated training work."""
        name = payload["name"] or f"checkpoint-{payload['seq_id']:06d}"
        validate_checkpoint_segment(name)
        checkpoint_dir = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "weights", name)
        if not payload["overwrite"] and os.path.lexists(checkpoint_dir):
            raise UserInputError(f"checkpoint {name!r} already exists; pass overwrite=True to replace it")
        root = os.path.dirname(checkpoint_dir)
        written = await self.backend.save_slot(
            slot=record.slot,
            root=root,
            name=name,
            tenant_digest=tenant_digest(record.tenant),
            parametrization_fields=build_parametrization_fields(record, self.config),
        )
        tinker_path = f"tinker://{record.model_id}/weights/{name}"
        # an overwrite's old expiry names the old commit, so it never deletes this one, whether or not the commit lands
        try:
            # the commit's recovery walk and fsyncs hit shared storage; keep the event loop serving
            await asyncio.to_thread(self.backend.commit_adapter, root, name, written, overwrite=payload["overwrite"])
        except ValueError as error:
            # the commit refuses to replace a link it did not write (e.g. an operator's link to a shared adapter)
            raise UserInputError(f"cannot commit checkpoint {name!r}: {error}") from None
        except OSError as error:
            # a storage failure loses only this save
            return {"error": f"cannot commit checkpoint {name!r}: {error}", "error_category": "server"}
        self._delete_retired_outside_the_lock(root)
        await self._set_expiry_after_commit(tinker_path, written.commit_id, payload["ttl_seconds"])
        return {"op": "save_state", "path": tinker_path}

    async def _set_expiry_after_commit(self, tinker_path: str, commit_id: str, ttl_seconds: int | None) -> None:
        try:
            async with self._expiry_lock:
                # two fsyncs on shared storage
                await asyncio.to_thread(self.expiry.set, tinker_path, commit_id, ttl_seconds, time.time())
        except OSError:
            # failing the request would hide a committed save; without its entry it only does not expire
            logger.exception(f"could not record the expiry of {tinker_path}; it does not expire")

    def _delete_retired_outside_the_lock(self, root: str) -> None:
        task = asyncio.create_task(self._delete_retired(root))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        task.add_done_callback(self._observe_background_task)

    async def _delete_retired(self, root: str) -> None:
        async with self._cleanup_lock:
            try:
                await asyncio.to_thread(self.backend.delete_retired_adapters, root)
            except OSError:
                # best effort after a committed save: the .deleting dirs go on the next pass or at startup
                logger.exception(f"could not delete retired adapter dirs under {root}")

    async def _load_state(self, record: ModelRecord, payload: dict) -> dict:
        source_id, kind, name = parse_tinker_path(payload["path"])
        if kind != "weights":
            raise UserInputError(f"cannot load {kind} into a training model; use a save_state checkpoint")
        checkpoint_dir = resolve_checkpoint_dir(self.config.checkpoint_root, source_id, kind, name)
        source_tenant = payload.get("weights_access_token")
        if source_tenant is None:
            source_tenant = record.tenant
        committed = self.backend.resolve_adapter(os.path.dirname(checkpoint_dir), name, source_tenant, source_id)
        if committed is None:
            # saved before commits: torch_dist with META.json, which the trainer's legacy loader reads
            source, ckpt_path = (
                read_checkpoint_metadata(checkpoint_dir, source_tenant, payload["path"]),
                checkpoint_dir,
            )
        else:
            source, ckpt_path = committed["parametrization"], committed["path"]
        # the trainer checks the rest of the parametrization against the model
        for key, expected in (("base_model", record.base_model), ("lora_rank", record.lora_rank)):
            if source[key] != expected:
                raise UserInputError(
                    f"checkpoint {payload['path']!r} has {key}={source[key]!r}; this model has {expected!r}"
                )
        fields = build_parametrization_fields(record, self.config) | {"lora_alpha": source["lora_alpha"]}
        failure = await self.backend.load_slot(
            record.slot,
            record.lora_rank,
            source["lora_alpha"],
            source={
                "path": os.path.realpath(ckpt_path),
                "load_optimizer": payload["optimizer"],
                "parametrization_fields": fields,
            },
        )
        if failure is not None:
            return failure
        # the tenant record owns alpha; the SDK cannot set it, so a loaded adapter keeps its own
        record.lora_alpha = source["lora_alpha"]
        return {"op": "load_state"}

    async def _save_weights_for_sampler(self, record: ModelRecord, payload: dict) -> dict:
        version = payload.get("sampler_path")
        if version is None:
            version = str(record.next_sampler_version)
            record.next_sampler_version += 1
        else:
            validate_checkpoint_segment(version)
        path = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "sampler_weights", version)
        if os.path.lexists(path):
            raise UserInputError(f"sampler weights {version!r} already exist; save under a new name")
        root = os.path.dirname(path)
        # the tenant's key stays on the gateway; the trainer ranks see only its digest
        written = await self.backend.export_slot(
            slot=record.slot,
            root=root,
            name=version,
            tenant_digest=tenant_digest(record.tenant),
            parametrization_fields=build_parametrization_fields(record, self.config),
        )
        # the commit's recovery walk and fsyncs hit shared storage; keep the event loop serving
        try:
            await asyncio.to_thread(self.backend.commit_adapter, root, version, written, overwrite=False)
        except ValueError as error:
            # the commit refuses to replace a link it did not write (e.g. an operator's link to a shared adapter)
            raise UserInputError(f"cannot commit sampler weights {version!r}: {error}") from None
        except OSError as error:
            # a storage failure loses only this export; the trainer and other tenants are unaffected
            return {"error": f"cannot commit sampler weights {version!r}: {error}", "error_category": "server"}
        result = {
            "op": "save_weights_for_sampler",
            "path": f"tinker://{record.model_id}/sampler_weights/{version}",
        }
        await self._set_expiry_after_commit(result["path"], written.commit_id, payload["ttl_seconds"])
        if payload.get("sampler_path") is None:
            # unnamed saves return a sampling session bound to the new version
            result["sampling_session_id"] = self._new_sampling_session(
                record.tenant, record.session_id, result["path"]
            )
        return result

    async def _save_weights_external(self, record: ModelRecord, payload: dict) -> dict:
        """The official Tinker export of the adapter, for download (get_external_weights_urls)."""
        if not self.config.exports_tinker_names:
            raise UserInputError(
                "this model family has no checked official Tinker names; save_weights_external is unsupported"
            )
        name = payload["name"]
        validate_checkpoint_segment(name)
        path = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "external_weights", name)
        if os.path.lexists(path):
            raise UserInputError(f"external weights {name!r} already exist; save under a new name")
        root = os.path.dirname(path)
        written = await self.backend.export_external(
            slot=record.slot,
            root=root,
            name=name,
            tenant_digest=tenant_digest(record.tenant),
            parametrization_fields=build_parametrization_fields(record, self.config),
            age_encryption_recipients=payload["age_encryption_recipients"],
        )
        try:
            await asyncio.to_thread(self.backend.commit_adapter, root, name, written, overwrite=False)
        except OSError as error:
            # e.g. a full disk: the trainer is healthy, so only this save fails
            return {"error": f"could not commit external weights {name!r}: {error}", "error_category": "server"}
        tinker_path = f"tinker://{record.model_id}/external_weights/{name}"
        await self._set_expiry_after_commit(tinker_path, written.commit_id, payload["ttl_seconds"])
        try:
            size = await asyncio.to_thread(self._checkpoint_size, root, name, record.tenant)
        except OSError:
            # the export is committed; the SDK takes an unknown size
            logger.exception(f"could not size {tinker_path}")
            size = None
        return {"op": "save_weights_external", "path": tinker_path, "size_bytes": size}

    def _checkpoint_size(self, root: str, name: str, tenant: str) -> int:
        files = self.backend.resolve_checkpoint_files(root, name, tenant)
        return sum(os.path.getsize(os.path.join(files["path"], f)) for f in files["files"])

    def _owned_checkpoint(self, tenant: str, model_id: str, checkpoint_id: str) -> tuple[str, str, str, dict]:
        """(tinker path, its adapter root, name, its files) of a committed checkpoint ``tenant`` owns."""
        tinker_path = f"tinker://{model_id}/{checkpoint_id}"
        _, kind, name = parse_tinker_path(tinker_path)
        root = os.path.dirname(resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name))
        try:
            files = self.backend.resolve_checkpoint_files(root, name, tenant)
        except ValueError as error:
            # a name linked to a dir this gateway did not write (resolve_adapter_dir refuses it)
            raise UserInputError(str(error)) from None
        if files is None:
            if os.path.isdir(os.path.join(root, name)):
                raise UserInputError(
                    f"checkpoint {tinker_path} was saved before checkpoints could expire or be deleted through the API"
                )
            raise NotFoundError(f"checkpoint {tinker_path} does not exist")
        return tinker_path, root, name, files

    async def external_weights_urls(self, tenant: str, model_id: str, checkpoint_id: str, base_url: str) -> dict:
        """Download URLs that carry no credential, bound to this tenant, the checkpoint and an expiry."""
        if not checkpoint_id.startswith("external_weights/"):
            raise UserInputError(f"tinker://{model_id}/{checkpoint_id} is not an external_weights checkpoint")
        tinker_path, _, _, files = await asyncio.to_thread(self._owned_checkpoint, tenant, model_id, checkpoint_id)
        now = time.time()
        expires_at = now + _DOWNLOAD_URL_TTL_S
        if (checkpoint_expires_at := self.expiry.expires_at(tinker_path, files["commit_id"])) is not None:
            # expired, and only not yet swept
            if checkpoint_expires_at <= now:
                raise NotFoundError(f"checkpoint {tinker_path} has expired")
            expires_at = min(expires_at, checkpoint_expires_at)
        key = (tenant, tinker_path, files["commit_id"])
        # a token keeps the expiry it was issued with; reuse it while it has half the life a new one would get, so a
        # token capped by its checkpoint's expiry is reused too
        live = self._downloads.get(self._download_tokens.get(key))
        if live is not None and live.expires_at - now >= (expires_at - now) / 2:
            token, expires_at = self._download_tokens[key], live.expires_at
        else:
            token = self._download_tokens[key] = secrets.token_urlsafe(32)
            self._downloads[token] = _Download(*key, expires_at)
        return {
            "urls": {f: f"{base_url}api/v1/external_weights/{token}/{f}" for f in files["files"]},
            "expires": datetime.fromtimestamp(expires_at, UTC).isoformat(),
        }

    async def download_path(self, token: str, filename: str) -> str:
        """The file a download URL names; gone once the URL expires or the checkpoint is deleted."""
        download = self._downloads.get(token)
        if download is None or download.expires_at <= time.time():
            raise NotFoundError("unknown or expired download URL")
        model_id, kind, name = parse_tinker_path(download.tinker_path)
        try:
            _, _, _, files = await asyncio.to_thread(
                self._owned_checkpoint, download.tenant, model_id, f"{kind}/{name}"
            )
        except NotFoundError:
            raise NotFoundError(f"{download.tinker_path} was deleted") from None
        if files["commit_id"] != download.commit_id:
            raise NotFoundError(f"{download.tinker_path} was deleted and saved again; sign a new URL")
        if filename not in files["files"]:
            raise NotFoundError(f"{download.tinker_path} has no file {filename!r}")
        return os.path.join(files["path"], filename)

    async def set_checkpoint_ttl(self, tenant: str, model_id: str, checkpoint_id: str, ttl_seconds) -> None:
        # with deletes, so a TTL never outlives its checkpoint onto a later save of the same name
        async with self._expiry_lock:
            tinker_path, _, _, files = await asyncio.to_thread(self._owned_checkpoint, tenant, model_id, checkpoint_id)
            await asyncio.to_thread(self.expiry.set, tinker_path, files["commit_id"], ttl_seconds, time.time())

    async def delete_checkpoint(self, tenant: str, model_id: str, checkpoint_id: str) -> None:
        async with self._trainer_lock, self._expiry_lock:
            tinker_path, root, name, _ = await asyncio.to_thread(
                self._owned_checkpoint, tenant, model_id, checkpoint_id
            )
            await self._delete_checkpoint(tinker_path, root, name, tenant)

    async def _delete_checkpoint(self, tinker_path: str, root: str, name: str, tenant: str | None) -> None:
        """Requires the trainer and expiry locks; ``tenant`` None skips the ownership check (expiry)."""
        await asyncio.to_thread(self.backend.delete_adapter, root, name, tenant)
        await asyncio.to_thread(self.expiry.drop, tinker_path)
        self._delete_retired_outside_the_lock(root)
        logger.info(f"deleted checkpoint {tinker_path}")

    async def _run_expiry_sweeper(self) -> None:
        while True:
            await asyncio.sleep(_EXPIRY_SWEEP_S)
            self._drop_expired_downloads(time.time())
            await self._expire_checkpoints()

    def _drop_expired_downloads(self, now: float) -> None:
        for token, download in list(self._downloads.items()):
            if download.expires_at <= now:
                del self._downloads[token]
                # the key may already name a newer token
                if self._download_tokens.get(download[:3]) == token:
                    del self._download_tokens[download[:3]]

    async def _expire_checkpoints(self) -> None:
        # the trainer lock waits for a whole training call; take it only when there is something to delete
        if not self.expiry.expired(time.time()):
            return
        async with self._trainer_lock, self._expiry_lock:
            for tinker_path, entry in self.expiry.expired(time.time()):
                model_id, kind, name = parse_tinker_path(tinker_path)
                try:
                    root = os.path.dirname(resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name))
                    files = await asyncio.to_thread(self.backend.resolve_checkpoint_files, root, name, None)
                    if files is None or files["commit_id"] != entry.commit_id:
                        # deleted by hand, or the expiry of a commit an overwrite replaced: nothing to remove
                        await asyncio.to_thread(self.expiry.drop, tinker_path)
                    else:
                        await self._delete_checkpoint(tinker_path, root, name, None)
                except (OSError, ValueError, AssertionError):
                    # one undeletable checkpoint must not stop the sweep; its entry stays for the next one
                    logger.exception(f"could not delete expired checkpoint {tinker_path}; retrying next sweep")

    def weights_info(self, tenant: str, tinker_path: str) -> dict:
        """What the SDK needs to rebuild a training client from a checkpoint."""
        model_id, kind, name = parse_tinker_path(tinker_path)
        if kind == "external_weights":
            raise UserInputError(f"{tinker_path} is a download; create a training client from a weights checkpoint")
        checkpoint_dir = resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name)
        if kind == "sampler_weights":
            committed = resolve_sampler_export(
                checkpoint_dir, tenant, model_id, tinker_path, self.backend.resolve_adapter
            )
        else:
            # a training checkpoint's link may point at a legacy dir an interrupted overwrite adopted
            committed = self.backend.resolve_adapter(os.path.dirname(checkpoint_dir), name, tenant, model_id)
        if committed is not None:
            parametrization = committed["parametrization"]
            # an adapter dir records targets, not group flags; any gateway or import may have written it
            return {
                "base_model": parametrization["base_model"],
                "is_lora": True,
                "lora_rank": parametrization["lora_rank"],
                **self.backend.lora_groups(parametrization["target_modules"]),
            }
        meta = read_checkpoint_metadata(checkpoint_dir, tenant, tinker_path)
        return {
            "base_model": meta["base_model"],
            "is_lora": True,
            "lora_rank": meta["lora_rank"],
            "train_attn": meta["train_attn"],
            "train_mlp": meta["train_mlp"],
            "train_unembed": meta["train_unembed"],
        }

    def create_sampling_session(self, tenant: str, payload: dict) -> str:
        session = self._session_for(tenant, payload["session_id"])
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        seq_id = validate_seq_id(payload["sampling_session_seq_id"], "sampling_session_seq_id", minimum=0)
        if (previous := session.sampling_sessions_by_seq.get(seq_id)) is not None:
            return previous
        sampling_session_id = self._new_sampling_session(tenant, payload["session_id"], payload.get("model_path"))
        session.sampling_sessions_by_seq[seq_id] = sampling_session_id
        return sampling_session_id

    def _new_sampling_session(self, tenant: str, session_id: str, model_path: str | None) -> str:
        if model_path is not None:
            resolve_sampler_checkpoint(self.config, tenant, model_path, self.backend.resolve_adapter)
        sampling_session_id = f"sampling-{uuid.uuid4().hex}"
        self.sampling_sessions[sampling_session_id] = SamplingSessionRecord(
            tenant=tenant, model_path=model_path, session_id=session_id
        )
        return sampling_session_id

    def get_sampler(self, tenant: str, sampling_session_id: str) -> dict:
        session = self.sampling_sessions.get(sampling_session_id)
        if session is None:
            raise UserInputError(f"unknown sampling session {sampling_session_id!r}")
        if session.tenant != tenant:
            raise OwnershipError("sampling session does not belong to this tenant")
        return {
            "sampler_id": sampling_session_id,
            "base_model": self.config.base_model,
            "model_path": session.model_path,
        }

    def submit_sample(self, tenant: str, payload: dict) -> tuple[str, list[str]]:
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        model_path = payload.get("model_path")
        sampling_session = None
        if payload.get("sampling_session_id"):
            sampling_session_id = payload["sampling_session_id"]
            sampling_session = self.sampling_sessions.get(sampling_session_id)
            if sampling_session is None:
                raise UserInputError(
                    f"unknown sampling session {sampling_session_id!r}; create a sampling session first"
                )
            if sampling_session.tenant != tenant:
                raise OwnershipError("sampling session does not belong to this tenant")
            model_path = model_path or sampling_session.model_path
            seq_id = validate_seq_id(payload["seq_id"], "seq_id", minimum=0)
            if (previous := sampling_session.samples_by_seq.get(seq_id)) is not None:
                request_id, sequence_ids = previous
                request_id = self.futures.request_id_for_retry(request_id, model_path or "base", tenant)
                sampling_session.samples_by_seq[seq_id] = (request_id, sequence_ids)
                return request_id, sequence_ids
        validate_sample_payload(payload, self.config)
        params = payload["sampling_params"]
        # the shared sampling/training window takes precedence over the requested generation maximum
        payload = payload | {
            "sampling_params": params
            | {
                "max_tokens": min(
                    params["max_tokens"], self.config.max_tokens_per_sample - len(payload["prompt_tokens"])
                )
            }
        }
        lora_name, lora_path = (
            resolve_sampler_checkpoint(self.config, tenant, model_path, self.backend.resolve_adapter)
            if model_path
            else (None, None)
        )
        future = self.futures.create(model_path or "base", tenant)
        sequence_ids = [f"seq-{uuid.uuid4().hex}" for _ in range(payload.get("num_samples", 1))]
        task = asyncio.create_task(self._run_sample(future.request_id, sequence_ids, payload, lora_name, lora_path))
        session_id = sampling_session.session_id if sampling_session is not None else None
        self._sample_tasks[future.request_id] = (task, session_id)
        task.add_done_callback(lambda _t, rid=future.request_id: self._sample_tasks.pop(rid, None))
        task.add_done_callback(self._observe_background_task)
        if sampling_session is not None:
            sampling_session.samples_by_seq[seq_id] = (future.request_id, sequence_ids)
        return future.request_id, sequence_ids

    async def _run_sample(
        self,
        request_id: str,
        sequence_ids: list[str],
        payload: dict,
        lora_name: str | None,
        lora_path: str | None = None,
    ) -> None:
        try:
            result = await self.backend.sample(payload, lora_name, lora_path)
        except asyncio.CancelledError:
            self.futures.fail(request_id, "cancelled", "user")
        except UserInputError as error:
            self.futures.fail(request_id, str(error), "user")
        else:
            if "error" in result:
                self.futures.fail(request_id, result["error"], "server")
            else:
                for sequence_id, sequence in zip(sequence_ids, result["sequences"], strict=True):
                    sequence["sequence_id"] = sequence_id
                self.futures.resolve(request_id, {"op": "sample", **result})

    def cancel(self, tenant: str, request_id: str) -> None:
        """Cancel an in-flight sampling future; training commands have no
        cancel in the protocol and are ignored."""
        if self.futures.get(request_id, tenant) is None:
            return
        entry = self._sample_tasks.get(request_id)
        if entry is not None:
            entry[0].cancel()

    async def _run_lease_sweeper(self) -> None:
        """Reclaim from stale sessions: cancel sampling, unload models, free
        slots. Training state dies with the lease; only checkpoints survive."""
        while True:
            await asyncio.sleep(30)
            await self._expire_sessions()

    async def _expire_sessions(self) -> None:
        async with self._trainer_lock:
            now = time.monotonic()
            expired_sessions = {
                session_id
                for session_id, session in self.sessions.items()
                if now - session.last_heartbeat >= self.config.lease_timeout_s
            }
            await self._end_sessions(expired_sessions, "lease expired", lease_expired=True)
            # a client silent for longer than a lease would have lost the session anyway
            retention_s = max(self.config.lease_timeout_s, _SDK_FINISH_RETRY_S)
            for session_id, ended in list(self._ended_sessions.items()):
                if now - ended.ended_at >= retention_s:
                    del self._ended_sessions[session_id]

    async def finish_session(self, tenant: str, session_id: str, reason: str, detail: str | None) -> bool:
        """SDK close(): reclaim the session now instead of at lease expiry. False if its lease already expired.

        The first finish wins; a repeat (an SDK retry) is accepted without effect.
        """
        async with self._trainer_lock:
            ended = self._ended_sessions.get(session_id)
            if ended is not None and ended.tenant == tenant:
                return not ended.lease_expired
            self._session_for(tenant, session_id)
            logger.info(f"session {session_id} finished ({reason}): {detail}")
            await self._end_sessions({session_id}, f"session finished ({reason})", lease_expired=False)
            return True

    async def _end_sessions(self, session_ids: set[str], error: str, *, lease_expired: bool) -> None:
        """Drop sessions with their sampling sessions and models; requires the trainer lock."""
        now = time.monotonic()
        for session_id in session_ids:
            self._ended_sessions[session_id] = _EndedSession(self.sessions.pop(session_id).tenant, lease_expired, now)
        for sampling_session_id, record in list(self.sampling_sessions.items()):
            if record.session_id in session_ids:
                del self.sampling_sessions[sampling_session_id]

        for request_id, (task, session_id) in list(self._sample_tasks.items()):
            if session_id in session_ids:
                logger.warning(f"{error} for session of sample {request_id}; cancelling")
                self.futures.fail(request_id, error, "user")
                task.cancel()
        for model_id, record in list(self.models.items()):
            if record.session_id in session_ids:
                logger.warning(f"{error} for {model_id}; freeing slot {record.slot}")
                await self._close_model(model_id, error, "user")
