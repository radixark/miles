"""Sessions, models, futures, and ordered trainer dispatch.

The trainer lock serializes trainer calls across dispatch, model creation, and lease expiry."""

import asyncio
import logging
import os
import time
import uuid

from miles.tinker.core.future import RequestFuture, RequestFutureStore
from miles.tinker.core.input_validation import (
    validate_batch_payload,
    validate_checkpoint_compatibility,
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
    OwnershipError,
    SamplingSessionRecord,
    SessionRecord,
    UserInputError,
)
from miles.tinker.core.utils import (
    build_checkpoint_metadata,
    parse_tinker_path,
    read_checkpoint_metadata,
    resolve_checkpoint_dir,
    resolve_sampler_checkpoint,
)

logger = logging.getLogger(__name__)


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
        self._arrival_counter = 0
        self._batch_counter = 0
        self._background_error: BaseException | None = None
        # why each model closed, so later requests get the reason instead of "unknown model"
        self._close_reasons: dict[str, str] = {}

    async def run(self) -> None:
        sweep_task = asyncio.create_task(self._run_lease_sweeper())
        sweep_task.add_done_callback(self._observe_background_task)
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
                        if self.backend.trainer_dead():
                            raise RuntimeError("the trainer workers died; exiting so clients get refused connections")
                        continue
                await self._wake.wait()
                self._wake.clear()
        finally:
            tasks = [sweep_task, *self._create_tasks, *(task for task, _ in self._sample_tasks.values())]
            for task in tasks:
                task.cancel()
            # Cleanup failures must not replace the error that stopped the dispatcher.
            await asyncio.gather(*tasks, return_exceptions=True)

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
        model_seq_id = validate_seq_id(payload["model_seq_id"], "model_seq_id")
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
        if self.backend.trainer_dead():
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
            if pending.command.op.requires_model_close_on_failure():
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
        if not payload["overwrite"] and os.path.exists(checkpoint_dir):
            raise UserInputError(f"checkpoint {name!r} already exists; pass overwrite=True to replace it")
        if os.path.isdir(checkpoint_dir) and not os.path.islink(checkpoint_dir):
            raise UserInputError(f"cannot overwrite legacy checkpoint {name!r}; save under a new name")
        failure = await self.backend.save_slot(
            record.slot, checkpoint_dir, metadata=build_checkpoint_metadata(record, self.config)
        )
        if failure is not None:
            return failure
        return {"op": "save_state", "path": f"tinker://{record.model_id}/weights/{name}"}

    async def _load_state(self, record: ModelRecord, payload: dict) -> dict:
        source_id, kind, name = parse_tinker_path(payload["path"])
        if kind != "weights":
            raise UserInputError("cannot load sampler weights into a training model; use a save_state checkpoint")
        checkpoint_dir = os.path.realpath(resolve_checkpoint_dir(self.config.checkpoint_root, source_id, kind, name))
        source_tenant = payload.get("weights_access_token")
        if source_tenant is None:
            source_tenant = record.tenant
        meta = read_checkpoint_metadata(checkpoint_dir, source_tenant, payload["path"])
        validate_checkpoint_compatibility(meta, record, self.config, payload["path"])
        failure = await self.backend.load_slot(
            record.slot,
            record.lora_rank,
            record.lora_alpha,
            ckpt_path=checkpoint_dir,
            load_optimizer=payload["optimizer"],
        )
        if failure is not None:
            return failure
        return {"op": "load_state"}

    async def _save_weights_for_sampler(self, record: ModelRecord, payload: dict) -> dict:
        version = payload.get("sampler_path")
        if version is None:
            version = str(record.next_sampler_version)
            record.next_sampler_version += 1
        else:
            validate_checkpoint_segment(version)
        path = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "sampler_weights", version)
        if os.path.exists(path):
            raise UserInputError(f"sampler weights {version!r} already exist; save under a new name")
        failure = await self.backend.export_slot(
            record.slot,
            record.lora_rank,
            record.lora_alpha,
            path,
            metadata=build_checkpoint_metadata(record, self.config),
        )
        if failure is not None:
            return failure
        result = {
            "op": "save_weights_for_sampler",
            "path": f"tinker://{record.model_id}/sampler_weights/{version}",
        }
        if payload.get("sampler_path") is None:
            # unnamed saves return a sampling session bound to the new version
            result["sampling_session_id"] = self._new_sampling_session(
                record.tenant, record.session_id, result["path"]
            )
        return result

    def weights_info(self, tenant: str, tinker_path: str) -> dict:
        """What the SDK needs to rebuild a training client from a checkpoint."""
        model_id, kind, name = parse_tinker_path(tinker_path)
        meta = read_checkpoint_metadata(
            resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name), tenant, tinker_path
        )
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
        seq_id = validate_seq_id(payload["sampling_session_seq_id"], "sampling_session_seq_id")
        if (previous := session.sampling_sessions_by_seq.get(seq_id)) is not None:
            return previous
        sampling_session_id = self._new_sampling_session(tenant, payload["session_id"], payload.get("model_path"))
        session.sampling_sessions_by_seq[seq_id] = sampling_session_id
        return sampling_session_id

    def _new_sampling_session(self, tenant: str, session_id: str, model_path: str | None) -> str:
        if model_path is not None:
            resolve_sampler_checkpoint(self.config.checkpoint_root, tenant, model_path, self.config.base_model)
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
            seq_id = validate_seq_id(payload["seq_id"], "seq_id")
            if (previous := sampling_session.samples_by_seq.get(seq_id)) is not None:
                request_id, sequence_ids = previous
                request_id = self.futures.request_id_for_retry(request_id, model_path or "base", tenant)
                sampling_session.samples_by_seq[seq_id] = (request_id, sequence_ids)
                return request_id, sequence_ids
        validate_sample_payload(payload, self.config)
        lora_name, lora_path = (
            resolve_sampler_checkpoint(self.config.checkpoint_root, tenant, model_path, self.config.base_model)
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
            for session_id in expired_sessions:
                del self.sessions[session_id]
            for sampling_session_id, record in list(self.sampling_sessions.items()):
                if record.session_id in expired_sessions:
                    del self.sampling_sessions[sampling_session_id]

            for request_id, (task, session_id) in list(self._sample_tasks.items()):
                if session_id in expired_sessions:
                    logger.warning(f"lease expired for session of sample {request_id}; cancelling")
                    self.futures.fail(request_id, "lease expired", "user")
                    task.cancel()
            for model_id, record in list(self.models.items()):
                if record.session_id in expired_sessions:
                    logger.warning(f"lease expired for {model_id}; freeing slot {record.slot}")
                    await self._close_model(model_id, "lease expired", "user")
