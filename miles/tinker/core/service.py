"""Sessions, models, futures, and ordered trainer dispatch.

The backend lock serializes trainer calls across dispatch, model creation, and lease expiry."""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import suppress
from pathlib import Path

from miles.tinker.core.future import Future, FutureStore
from miles.tinker.core.planner import BarrierUnit, BatchUnit, Planner
from miles.tinker.core.stream import ModelStream
from miles.tinker.core.types import (
    LOSS_FN_INPUTS,
    LOSS_INPUT_KEYS,
    Command,
    CommandOp,
    GatewayConfig,
    ModelRecord,
    OwnershipError,
    UserInputError,
)

logger = logging.getLogger(__name__)


class TinkerService:
    def __init__(self, backend, config: GatewayConfig) -> None:
        self.backend = backend
        self.config = config
        self.futures = FutureStore()
        self.planner = Planner(config.batch_token_budget)
        self.models: dict[str, ModelRecord] = {}
        self.sessions: dict[str, dict] = {}
        self.sampling_sessions: dict[str, dict] = {}
        self.free_slots = set(range(config.n_slots))
        self._wake = asyncio.Event()
        self._backend_lock = asyncio.Lock()
        self._sample_tasks: dict[str, tuple] = {}  # request_id -> (task, tenant)
        self._create_tasks: set = set()
        self._arrival_counter = 0
        self._batch_counter = 0
        self._background_error: BaseException | None = None
        # why each evicted model died, so later requests get the reason instead of "unknown model"
        self._eviction_reasons: dict[str, str] = {}

    def create_session(self, tenant: str) -> str:
        session_id = f"session-{uuid.uuid4().hex}"
        self.sessions[session_id] = {
            "tenant": tenant,
            "last_heartbeat": time.monotonic(),
            "models_by_seq": {},
            "sampling_sessions_by_seq": {},
        }
        return session_id

    def _session_for(self, tenant: str, session_id: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None or session["tenant"] != tenant:
            raise UserInputError(f"unknown session {session_id!r}; create a session first")
        return session

    def heartbeat(self, tenant: str, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None and session["tenant"] == tenant:
            session["last_heartbeat"] = time.monotonic()

    def create_model(self, tenant: str, payload: dict) -> tuple[str, str]:
        """Two-phase like every command: allocate now, initialize the slot behind the future."""
        session = self._session_for(tenant, payload["session_id"])
        model_seq_id = _validate_seq_id(payload["model_seq_id"], "model_seq_id", minimum=0)
        if (previous := session["models_by_seq"].get(model_seq_id)) is not None:
            return previous
        base_model = payload["base_model"]
        if base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        lora_config = payload.get("lora_config") or {}
        self._reject_unsupported_lora_config(lora_config)
        rank = lora_config.get("rank", 32)
        if rank > self.config.max_lora_rank:
            raise UserInputError(
                f"lora_config.rank={rank} exceeds this gateway's slot capacity (--lora-rank {self.config.max_lora_rank})"
            )
        alpha = self.config.lora_alpha if self.config.lora_alpha is not None else float(2 * rank)
        if not self.free_slots:
            raise UserInputError(f"no free adapter slots (capacity {self.config.n_slots})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)

        model_id = f"model-{uuid.uuid4().hex[:12]}"
        record = ModelRecord(
            model_id=model_id,
            tenant=tenant,
            slot=slot,
            base_model=base_model,
            lora_rank=rank,
            lora_alpha=alpha,
        )
        self.models[model_id] = record
        self.planner.add_stream(ModelStream(model_id, tenant, slot))
        future = self.futures.create(model_id, tenant)
        task = asyncio.create_task(self._run_create_model(future.request_id, record))
        self._create_tasks.add(task)
        task.add_done_callback(self._observe_background_task)
        session["models_by_seq"][model_seq_id] = (future.request_id, model_id)
        return future.request_id, model_id

    def _reject_unsupported_lora_config(self, lora_config: dict) -> None:
        """Reject per-model settings that conflict with the fixed server adapter layout."""
        if lora_config.get("seed") is not None:
            raise UserInputError("lora_config.seed is not supported: adapter initialization is not per-model seedable")
        layout = {
            "train_attn": self.config.trains_attn,
            "train_mlp": self.config.trains_mlp,
            "train_unembed": self.config.trains_unembed,
        }
        for field, layout_trains in layout.items():
            requested = lora_config.get(field)
            if requested is not None and requested != layout_trains:
                raise UserInputError(
                    f"lora_config.{field}={requested} conflicts with this gateway's adapter layout "
                    f"({field}={layout_trains}); the layout is fixed by --target-modules at server start"
                )

    def _observe_background_task(self, task: asyncio.Task) -> None:
        self._create_tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            if self._background_error is None:
                self._background_error = error
            self._wake.set()

    async def _run_create_model(self, request_id: str, record: ModelRecord) -> None:
        async with self._backend_lock:
            failure = await self.backend.load_slot(record.slot, record.lora_rank, record.lora_alpha)
            if failure is not None:
                self.futures.fail(request_id, failure["error"], "server")
                await self._evict_model(record.model_id, failure["error"], "server")
                return
        self.futures.resolve(request_id, {"op": "create_model", "model_id": record.model_id})

    def get_model(self, tenant: str, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            reason = self._eviction_reasons.get(model_id)
            if reason is not None:
                raise UserInputError(f"model {model_id!r} was unloaded: {reason}")
            raise UserInputError(f"unknown model {model_id!r}")
        if record.tenant != tenant:
            raise OwnershipError(f"model {model_id} does not belong to this tenant")
        return record

    # -------- client submit path --------

    def submit(self, tenant: str, op: str, payload: dict) -> str:
        """Admit a decoded command; content errors settle its future as a user failure."""
        try:
            op = CommandOp(op)
        except ValueError:
            raise UserInputError(f"unknown command op {op!r}") from None
        model_id = payload["model_id"]
        self.get_model(tenant, model_id)
        seq_id = _validate_seq_id(payload["seq_id"], "seq_id")
        stream = self.planner.stream(model_id)

        # retries must not accumulate gradients twice
        if seq_id in stream.request_id_by_seq:
            request_id = stream.request_id_by_seq[seq_id]
            if self.futures.get(request_id, tenant) is not None:
                return request_id
            # expired results must fail terminally without re-executing the command
            replacement = self.futures.create(model_id, tenant)
            self.futures.fail(replacement.request_id, "result expired after retention", "user")
            stream.request_id_by_seq[seq_id] = replacement.request_id
            return replacement.request_id

        future = self.futures.create(model_id, tenant)
        stream.request_id_by_seq[seq_id] = future.request_id
        self._arrival_counter += 1
        validation_error = payload.get("validation_error")
        if validation_error is None:
            try:
                self._validate_batch_payload(op, payload)
            except UserInputError as error:
                validation_error = str(error)
        stream.submit(
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

    def _validate_batch_payload(self, op: CommandOp, payload: dict) -> None:
        if not op.is_batch():
            return
        datums = payload["datums"]
        if not datums:
            raise UserInputError("forward_backward with no data")
        if len(datums) > self.config.max_datums_per_request:
            raise UserInputError(
                f"{len(datums)} datums exceeds max_datums_per_request={self.config.max_datums_per_request}"
            )
        required_inputs = LOSS_FN_INPUTS.get(payload["loss_fn"])
        if required_inputs is None:
            raise UserInputError(f"unknown loss_fn {payload['loss_fn']!r}; known: {sorted(LOSS_FN_INPUTS)}")
        total_tokens = 0
        for index, datum in enumerate(datums):
            if len(datum["tokens"]) > self.config.max_tokens_per_datum:
                raise UserInputError(
                    f"datum {index}: {len(datum['tokens'])} tokens exceeds {self.config.max_tokens_per_datum}"
                )
            total_tokens += len(datum["tokens"])
            for wire_key in required_inputs:
                values = datum.get(LOSS_INPUT_KEYS[wire_key])
                if values is None:
                    raise UserInputError(
                        f"datum {index}: loss_fn {payload['loss_fn']!r} needs loss_fn_inputs[{wire_key!r}]"
                    )
                if len(values) != datum["target_len"]:
                    raise UserInputError(
                        f"datum {index}: loss_fn_inputs[{wire_key!r}] has {len(values)} values "
                        f"for {datum['target_len']} target tokens"
                    )
            unread = [
                wire_key
                for wire_key, datum_key in LOSS_INPUT_KEYS.items()
                if wire_key not in required_inputs and datum_key in datum
            ]
            if unread:
                raise UserInputError(
                    f"datum {index}: loss_fn {payload['loss_fn']!r} does not read loss_fn_inputs {unread}"
                )
        if total_tokens > self.config.max_tokens_per_request:
            raise UserInputError(
                f"{total_tokens} tokens exceeds max_tokens_per_request={self.config.max_tokens_per_request}"
            )

    def retrieve_future(self, tenant: str, request_id: str) -> Future | None:
        """None -> the HTTP layer answers 410 and the SDK resubmits."""
        return self.futures.get(request_id, tenant)

    # -------- dispatch loop --------

    async def run(self) -> None:
        sweep_task = asyncio.create_task(self.sweep_leases())
        sweep_task.add_done_callback(self._observe_background_task)
        try:
            while True:
                if self._background_error is not None:
                    raise self._background_error
                # unit selection shares the critical section with execution, so
                # lease expiry cannot reclaim a stream between the two
                async with self._backend_lock:
                    rejections = self.planner.ready_rejections()
                    if rejections:
                        for stream, pending in rejections:
                            await self._settle_request(stream, pending, {
                                "error": pending.command.validation_error, "error_category": "user"
                            })
                        continue
                    unit = self.planner.next_to_run()
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
            sweep_task.cancel()
            with suppress(asyncio.CancelledError):
                await sweep_task

    async def _run_batch(self, batch: BatchUnit) -> None:
        # slot-contiguous order; outputs come back aligned to it
        refs = sorted(batch.datums, key=lambda ref: ref.stream.slot)
        slot_datums = [(ref.stream.slot, ref.datum) for ref in refs]
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
                self.futures.resolve(
                    request.command.request_id, {"op": request.command.op, "outputs": request.outputs}
                )
                ref.stream.finish(request)

    async def _fail_batch(self, batch: BatchUnit, error: str, category: str) -> None:
        if batch.op.changes_training_state():
            for model_id in sorted({ref.stream.model_id for ref in batch.datums}):
                await self._evict_model(model_id, _failed_stream_message(error), category)
            return
        requests = {ref.request.command.request_id: (ref.stream, ref.request) for ref in batch.datums}
        for stream, pending in requests.values():
            await self._settle_request(stream, pending, {"error": error, "error_category": category})

    async def _run_barrier(self, barrier: BarrierUnit) -> None:
        try:
            outcomes = await self._dispatch_barrier_op(barrier)
        except (UserInputError, OwnershipError) as error:
            outcomes = [{"error": str(error), "error_category": "user"} for _ in barrier.entries]
        for (stream, pending), outcome in zip(barrier.entries, outcomes, strict=True):
            await self._settle_request(stream, pending, outcome)

    async def _settle_request(self, stream, pending, outcome: dict) -> None:
        if "error" in outcome:
            category = outcome.get("error_category", "server")
            if pending.command.op.changes_training_state():
                await self._evict_model(stream.model_id, _failed_stream_message(outcome["error"]), category)
                return
            self.futures.fail(pending.command.request_id, outcome["error"], category)
        else:
            self.futures.resolve(pending.command.request_id, outcome)
        stream.finish(pending)

    async def _dispatch_barrier_op(self, barrier: BarrierUnit) -> list[dict]:
        """Return one result or error per entry; only the caller settles futures and retires models."""
        if barrier.op == CommandOp.OPTIM_STEP:
            return await self._step_optimizers(barrier.entries)
        ((stream, pending),) = barrier.entries  # every other barrier is single-entry
        record = self.models[stream.model_id]
        payload = pending.command.payload
        if barrier.op == CommandOp.SAVE_STATE:
            return [await self._save_state(record, pending, payload)]
        if barrier.op == CommandOp.LOAD_STATE:
            return [await self._load_state(record, payload)]
        if barrier.op == CommandOp.SAVE_WEIGHTS_FOR_SAMPLER:
            return [await self._save_weights_for_sampler(record, payload)]
        raise UserInputError(f"unknown barrier op {barrier.op!r}")

    async def _step_optimizers(self, entries: list) -> list[dict]:
        adam_params_by_slot = {stream.slot: pending.command.payload["adam_params"] for stream, pending in entries}
        slot_outcomes = await self.backend.optim_step(adam_params_by_slot)
        outcomes = []
        for stream, _ in entries:
            outcome = slot_outcomes[stream.slot]
            if "error" in outcome:
                outcomes.append(outcome)
            else:
                outcomes.append({"op": "optim_step", "metrics": {key: float(value) for key, value in outcome.items()}})
        return outcomes


    async def _save_state(self, record: ModelRecord, pending, payload: dict) -> dict:
        """Save parameters and optimizer state; call after optim_step to persist accumulated training work."""
        name = payload["name"] or f"checkpoint-{pending.command.seq_id:06d}"
        _validate_checkpoint_segment(name)
        checkpoint_dir = self._checkpoint_dir(record.model_id, "weights", name)
        if not payload["overwrite"] and os.path.exists(checkpoint_dir):
            raise UserInputError(f"checkpoint {name!r} already exists; pass overwrite=True to replace it")
        if (failure := await self.backend.save_slot(record.slot, checkpoint_dir)) is not None:
            return failure
        if (failure := self._stamp_checkpoint_meta(checkpoint_dir, record)) is not None:
            return failure
        return {"op": "save_state", "path": f"tinker://{record.model_id}/weights/{name}"}

    async def _load_state(self, record: ModelRecord, payload: dict) -> dict:
        source_id, kind, name = _parse_tinker_path(payload["path"])
        meta = self._checkpoint_meta(self._checkpoint_dir(source_id, kind, name), record.tenant, payload["path"])
        self._reject_checkpoint_mismatch(meta, record, payload["path"])
        failure = await self.backend.load_slot(
            record.slot, record.lora_rank, record.lora_alpha,
            ckpt_path=self._checkpoint_dir(source_id, kind, name), load_optimizer=payload["optimizer"],
        )
        if failure is not None:
            return failure
        return {"op": "load_state"}

    async def _save_weights_for_sampler(self, record: ModelRecord, payload: dict) -> dict:
        snapshot = await self._save_sampler_snapshot(record, payload.get("sampler_path"))
        if isinstance(snapshot, dict):
            return snapshot
        version, path = snapshot
        await self._warm_sampler_cache(record, version, path)
        result = {
            "op": "save_weights_for_sampler",
            "path": f"tinker://{record.model_id}/sampler_weights/{version}",
        }
        if payload.get("sampler_path") is None:
            # unnamed saves return a sampling session bound to the new version
            result["sampling_session_id"] = self._new_sampling_session(record.tenant, result["path"])
        return result

    async def _save_sampler_snapshot(self, record: ModelRecord, version: str | None) -> tuple[str, str] | dict:
        if version is None:
            version = str(record.next_sampler_version)
            record.next_sampler_version += 1
        else:
            _validate_checkpoint_segment(version)
        path = self._checkpoint_dir(record.model_id, "sampler_weights", version)
        if os.path.exists(os.path.join(path, "META.json")):
            # engines may already hold this name's bytes; saved versions are immutable
            raise UserInputError(f"sampler weights {version!r} already exist; save under a new name")
        if (failure := await self.backend.export_slot(record.slot, record.lora_rank, record.lora_alpha, path)) is not None:
            return failure
        if (failure := self._stamp_checkpoint_meta(path, record)) is not None:
            return failure
        record.published_sampler_versions.add(version)
        return version, path

    async def _warm_sampler_cache(self, record: ModelRecord, version: str, path: str) -> None:
        failure = await self.backend.push_slot(
            record.slot, f"{record.model_id}@{version}", record.lora_rank, record.lora_alpha, lora_path=path
        )
        if failure is not None:
            logger.warning("sampler snapshot %s will backfill from disk: %s", version, failure["error"])

    def _reject_checkpoint_mismatch(self, meta: dict, record: ModelRecord, shown_path: str) -> None:
        """The tensors only keep their meaning under the config that wrote them (alpha scales them,
        the target layout names them); a restore under different settings would be silent corruption."""
        expected = {
            "base_model": record.base_model,
            "lora_rank": record.lora_rank,
            "lora_alpha": record.lora_alpha,
            "train_attn": self.config.trains_attn,
            "train_mlp": self.config.trains_mlp,
            "train_unembed": self.config.trains_unembed,
        }
        for key, value in expected.items():
            if meta[key] != value:
                raise UserInputError(
                    f"checkpoint {shown_path!r} was saved with {key}={meta[key]!r}; this model expects {key}={value!r}"
                )

    def _stamp_checkpoint_meta(self, checkpoint_dir: str, record: ModelRecord) -> dict | None:
        """Mark completed tensor shards as a gateway checkpoint with persistent ownership and shape."""
        meta = {
            # the digest proves ownership without persisting the bearer credential itself
            "tenant_digest": _tenant_digest(record.tenant),
            "base_model": record.base_model,
            "lora_rank": record.lora_rank,
            "lora_alpha": record.lora_alpha,
            "train_attn": self.config.trains_attn,
            "train_mlp": self.config.trains_mlp,
            "train_unembed": self.config.trains_unembed,
        }
        try:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            (Path(checkpoint_dir) / "META.json.tmp").write_text(json.dumps(meta, indent=2))
            os.replace(Path(checkpoint_dir) / "META.json.tmp", Path(checkpoint_dir) / "META.json")
        except OSError as error:
            return {"error": str(error)}
        return None

    def _checkpoint_meta(self, checkpoint_dir: str, tenant: str, shown_path: str) -> dict:
        meta_file = Path(checkpoint_dir) / "META.json"
        if not meta_file.exists():
            raise UserInputError(f"unknown checkpoint {shown_path!r}")
        try:
            meta = json.loads(meta_file.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise UserInputError(f"cannot read checkpoint {shown_path!r}: {error}") from error
        if meta["tenant_digest"] != _tenant_digest(tenant):
            raise OwnershipError(f"checkpoint {shown_path!r} does not belong to this tenant")
        return meta

    def _checkpoint_dir(self, model_id: str, kind: str, name: str) -> str:
        root = os.path.realpath(self.config.checkpoint_root)
        path = os.path.realpath(f"{root}/{model_id}/{kind}/{name}")
        assert path.startswith(root + os.sep), f"checkpoint path {path!r} escapes {root!r}"
        return path

    # -------- sampling plane (future-based but never queues) --------

    def create_sampling_session(self, tenant: str, payload: dict) -> str:
        session = self._session_for(tenant, payload["session_id"])
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        seq_id = _validate_seq_id(payload["sampling_session_seq_id"], "sampling_session_seq_id", minimum=0)
        if (previous := session["sampling_sessions_by_seq"].get(seq_id)) is not None:
            return previous
        sampling_session_id = self._new_sampling_session(tenant, payload.get("model_path"))
        session["sampling_sessions_by_seq"][seq_id] = sampling_session_id
        return sampling_session_id

    def _new_sampling_session(self, tenant: str, model_path: str | None) -> str:
        sampling_session_id = f"sampling-{uuid.uuid4().hex}"
        self.sampling_sessions[sampling_session_id] = {
            "tenant": tenant,
            "model_path": model_path,
            "samples_by_seq": {},
        }
        return sampling_session_id

    def submit_sample(self, tenant: str, payload: dict) -> tuple[str, list[str]]:
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        model_path = payload.get("model_path")
        sampling_session = None
        if payload.get("sampling_session_id"):
            sampling_session = self.sampling_sessions[payload["sampling_session_id"]]
            if sampling_session["tenant"] != tenant:
                raise OwnershipError("sampling session does not belong to this tenant")
            model_path = model_path or sampling_session["model_path"]
            seq_id = _validate_seq_id(payload["seq_id"], "seq_id", minimum=0)
            if (previous := sampling_session["samples_by_seq"].get(seq_id)) is not None:
                return previous
        if payload.get("num_samples", 1) > self.config.max_samples_per_request:
            raise UserInputError(
                f"num_samples {payload['num_samples']} exceeds max_samples_per_request="
                f"{self.config.max_samples_per_request}"
            )
        lora_name, lora_path = self._resolve_sampler(tenant, model_path) if model_path else (None, None)
        future = self.futures.create(model_path or "base", tenant)
        sequence_ids = [f"seq-{uuid.uuid4().hex}" for _ in range(payload.get("num_samples", 1))]
        task = asyncio.create_task(self._run_sample(future.request_id, payload, lora_name, lora_path))
        self._sample_tasks[future.request_id] = (task, tenant)
        task.add_done_callback(lambda _t, rid=future.request_id: self._sample_tasks.pop(rid, None))
        task.add_done_callback(self._observe_background_task)
        if sampling_session is not None:
            sampling_session["samples_by_seq"][seq_id] = (future.request_id, sequence_ids)
        return future.request_id, sequence_ids

    async def _run_sample(
        self, request_id: str, payload: dict, lora_name: str | None, lora_path: str | None = None
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
                self.futures.resolve(request_id, {"op": "sample", **result})

    def cancel(self, tenant: str, request_id: str) -> None:
        """Cancel an in-flight sampling future; training commands have no
        cancel in the protocol and are ignored."""
        if self.futures.get(request_id, tenant) is None:
            return
        entry = self._sample_tasks.get(request_id)
        if entry is not None:
            entry[0].cancel()

    def _resolve_sampler(self, tenant: str, model_path: str) -> tuple[str, str]:
        """-> (engine lora_name, adapter dir): the request carries both, so the
        engine can backfill an evicted version from disk on its own."""
        model_id, kind, name = _parse_tinker_path(model_path)
        if kind != "sampler_weights":
            raise UserInputError(f"cannot sample from {model_path!r}: not a sampler_weights path")
        checkpoint_dir = self._checkpoint_dir(model_id, "sampler_weights", name)
        record = self.models.get(model_id)
        if record is not None:
            if record.tenant != tenant:
                raise OwnershipError(f"model {model_id} does not belong to this tenant")
            if name not in record.published_sampler_versions:
                raise UserInputError(f"unknown sampler version {name} for {model_id}")
        else:
            # the training lease is gone; the checkpoint on disk is the record
            self._checkpoint_meta(checkpoint_dir, tenant, model_path)
        return f"{model_id}@{name}", checkpoint_dir

    def weights_info(self, tenant: str, tinker_path: str) -> dict:
        """What the SDK needs to rebuild a training client from a checkpoint."""
        model_id, kind, name = _parse_tinker_path(tinker_path)
        meta = self._checkpoint_meta(self._checkpoint_dir(model_id, kind, name), tenant, tinker_path)
        return {
            "base_model": meta["base_model"],
            "is_lora": True,
            "lora_rank": meta["lora_rank"],
            "train_attn": meta["train_attn"],
            "train_mlp": meta["train_mlp"],
            "train_unembed": meta["train_unembed"],
        }

    async def sweep_leases(self) -> None:
        """Reclaim from stale tenants: cancel sampling, unload models, free
        slots. Training state dies with the lease; only checkpoints survive."""
        while True:
            await asyncio.sleep(30)
            await self._sweep_once()

    async def _sweep_once(self) -> None:
        now = time.monotonic()
        had_sessions = bool(self.sessions)
        fresh_tenants = {
            session["tenant"]
            for session in self.sessions.values()
            if now - session["last_heartbeat"] < self.config.lease_timeout_s
        }

        def lease_expired(tenant: str) -> bool:
            # with no sessions at all there is no lease to expire
            return had_sessions and tenant not in fresh_tenants

        for session_id, session in list(self.sessions.items()):
            if lease_expired(session["tenant"]):
                del self.sessions[session_id]
        for sampling_session_id, record in list(self.sampling_sessions.items()):
            if lease_expired(record["tenant"]):
                del self.sampling_sessions[sampling_session_id]

        for request_id, (task, tenant) in list(self._sample_tasks.items()):
            if lease_expired(tenant):
                logger.warning(f"lease expired for tenant of sample {request_id}; cancelling")
                task.cancel()
        for model_id, record in list(self.models.items()):
            if not lease_expired(record.tenant):
                continue
            logger.warning(f"lease expired for {model_id}; freeing slot {record.slot}")
            async with self._backend_lock:
                await self._evict_model(model_id, "lease expired", "user")

    async def _evict_model(self, model_id: str, error: str, category: str) -> None:
        """Free a model's slot and fail its pending requests; requires the backend lock. Idempotent."""
        record = self.models.pop(model_id, None)
        if record is None:
            return
        self._eviction_reasons[model_id] = error
        while len(self._eviction_reasons) > 4 * self.config.n_slots:
            self._eviction_reasons.pop(next(iter(self._eviction_reasons)))
        stream = self.planner.stream(model_id)
        self.planner.remove_stream(model_id)
        for request_id in stream.request_id_by_seq.values():
            if self.futures.get(request_id, record.tenant) is not None:
                self.futures.fail(request_id, error, category)
        if self.backend.trainer_dead():
            return
        failure = await self.backend.unload_slot(record.slot)
        if failure is not None:
            logger.error("slot %s remains unavailable after unload failed: %s", record.slot, failure["error"])
            return
        self.free_slots.add(record.slot)


def _validate_seq_id(value, name: str, minimum: int = 1) -> int:
    # stream seq_ids are 1-based (the watermark starts at 0); idempotency keys are 0-based
    if not isinstance(value, int) or value < minimum:
        raise UserInputError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _tenant_digest(tenant: str) -> str:
    return hashlib.sha256(tenant.encode()).hexdigest()


def _failed_stream_message(error: str) -> str:
    return f"training stream failed ({error}); create a new model and restore from a checkpoint"


def _parse_tinker_path(path: str) -> tuple[str, str, str]:
    if not path.startswith("tinker://"):
        raise UserInputError(f"not a tinker path: {path!r}")
    parts = path.removeprefix("tinker://").split("/")
    if len(parts) != 3 or parts[1] not in ("weights", "sampler_weights"):
        raise UserInputError(f"malformed tinker path: {path!r}")
    for segment in parts:
        _validate_checkpoint_segment(segment)
    return parts[0], parts[1], parts[2]


def _validate_checkpoint_segment(segment: str) -> None:
    """Reject client path segments that could escape the checkpoint root."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", segment) is None:
        raise UserInputError(f"invalid checkpoint path segment {segment!r}")
