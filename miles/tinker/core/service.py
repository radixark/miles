"""The gateway service: sessions, models, streams, futures, and the single
dispatch loop.

Speaks only the internal language: server/ hands it decoded commands and
renders its results; runtime.py turns units into trainer batches. All backend
calls go through one loop / one lock: the trainer is an SPMD domain and must
see a single totally ordered stream of batches and barriers.
"""

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path

from miles.tinker.core.future import PENDING, Future, FutureStore
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


class ExecutorBackend:
    """What runtime.py implements. Speaks datums and plain lists; core stays
    torch-free and miles-free."""

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        raise NotImplementedError

    async def unload_slot(self, slot: int) -> None:
        raise NotImplementedError

    async def forward_backward(
        self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict]:
        """slot_datums: slot-sorted [(slot, datum)]. Returns one
        {"loss": float, "logprobs": [float]} per datum, in order."""
        raise NotImplementedError

    async def forward_only(self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        raise NotImplementedError

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        """-> per-slot outcome: {"grad_norm": x} stepped, {"skipped_nonfinite": 1.0}
        dropped non-finite grads, {"error": msg} failed."""
        raise NotImplementedError

    async def zero_grads(self, slot: int) -> None:
        """Drop the slot's accumulated gradients."""
        raise NotImplementedError

    async def save_slot(self, slot: int, path: str) -> None:
        raise NotImplementedError

    async def export_slot(self, slot: int, rank: int, alpha: float, path: str) -> None:
        """Write the slot's adapter as an engine-loadable dir."""
        raise NotImplementedError

    async def push_slot(
        self, slot: int, lora_name: str, rank: int, alpha: float, lora_path: str | None = None
    ) -> None:
        raise NotImplementedError

    async def sample(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        """-> {"sequences": [{"tokens", "logprobs", "stop_reason"}],
        "prompt_logprobs"?, "topk_prompt_logprobs"?}"""
        raise NotImplementedError


class TinkerService:
    def __init__(self, backend: ExecutorBackend, config: GatewayConfig) -> None:
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
        # slot -> (error, category) of a discarded accumulation window; the next
        # optim barrier on the slot must fail instead of stepping ghost gradients
        self._poisoned_slots: dict[int, tuple[str, str]] = {}

    # -------- control plane --------

    def create_session(self, tenant: str, payload: dict) -> str:
        session_id = f"session-{uuid.uuid4().hex}"
        self.sessions[session_id] = {"tenant": tenant, "last_heartbeat": time.monotonic(), "payload": payload}
        return session_id

    def heartbeat(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            session["last_heartbeat"] = time.monotonic()

    def create_model(self, tenant: str, payload: dict) -> tuple[str, str]:
        """Two-phase like every command: allocate now, initialize the slot behind the future."""
        base_model = payload["base_model"]
        if base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        lora_config = payload.get("lora_config") or {}
        rank = lora_config.get("rank", 32)
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
            session_id=payload.get("session_id", ""),
            user_metadata=payload.get("user_metadata") or {},
        )
        self.models[model_id] = record
        self.planner.add_stream(ModelStream(model_id, tenant, slot))
        future = self.futures.create(model_id, tenant)
        task = asyncio.create_task(self._run_create_model(future.request_id, record))
        self._create_tasks.add(task)
        task.add_done_callback(self._create_tasks.discard)
        return future.request_id, model_id

    async def _run_create_model(self, request_id: str, record: ModelRecord) -> None:
        try:
            async with self._backend_lock:
                await self.backend.load_slot(record.slot, record.lora_rank, record.lora_alpha)
        except Exception as error:
            self.models.pop(record.model_id, None)
            self.planner.remove_stream(record.model_id)
            self.free_slots.add(record.slot)
            self.futures.fail(request_id, str(error), "server")
            return
        self.futures.resolve(request_id, {"op": "create_model", "model_id": record.model_id})

    def get_model(self, tenant: str, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            raise UserInputError(f"unknown model {model_id!r}")
        if record.tenant != tenant:
            raise OwnershipError(f"model {model_id} does not belong to this tenant")
        return record

    # -------- command plane --------

    def submit(self, tenant: str, op: str, payload: dict) -> str:
        """payload is server-decoded; content errors here are admission
        rejections and fail the future (the SDK sees RequestFailedError)."""
        try:
            op = CommandOp(op)
        except ValueError:
            raise UserInputError(f"unknown command op {op!r}") from None
        model_id = payload["model_id"]
        self.get_model(tenant, model_id)
        seq_id = payload["seq_id"]
        stream = self.planner.stream(model_id)

        # idempotency: the SDK resends the same seq_id after timeouts/410;
        # re-executing forward_backward would double-accumulate gradients
        if seq_id in stream.request_id_by_seq:
            request_id = stream.request_id_by_seq[seq_id]
            if self.futures.get(request_id, tenant) is not None:
                return request_id
            # the result aged out of retention; the command already executed, so
            # re-running it is unsafe — answer with a terminal failure instead of 410 forever
            replacement = self.futures.create(model_id, tenant)
            self.futures.fail(replacement.request_id, "result expired after retention", "user")
            stream.request_id_by_seq[seq_id] = replacement.request_id
            return replacement.request_id

        future = self.futures.create(model_id, tenant)
        stream.request_id_by_seq[seq_id] = future.request_id
        try:
            self._admit(op, payload)
        except UserInputError as error:
            self.futures.fail(future.request_id, str(error), "user")
            # the rejected command still consumes its seq position, or the
            # stream would wait for it forever
            payload = {**payload, "datums": []}

        self._arrival_counter += 1
        stream.submit(
            Command(
                model_id=model_id,
                seq_id=seq_id,
                op=op,
                payload=payload,
                request_id=future.request_id,
                arrival=self._arrival_counter,
            )
        )
        self._wake.set()
        return future.request_id

    def _admit(self, op: CommandOp, payload: dict) -> None:
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
        if total_tokens > self.config.max_tokens_per_request:
            raise UserInputError(
                f"{total_tokens} tokens exceeds max_tokens_per_request={self.config.max_tokens_per_request}"
            )

    def retrieve_future(self, tenant: str, request_id: str) -> Future | None:
        """None -> the HTTP layer answers 410 and the SDK resubmits."""
        return self.futures.get(request_id, tenant)

    # -------- sampling plane (future-based but never queues) --------

    def create_sampling_session(self, tenant: str, payload: dict) -> str:
        sampling_session_id = f"sampling-{uuid.uuid4().hex}"
        self.sampling_sessions[sampling_session_id] = {
            "tenant": tenant,
            "base_model": payload.get("base_model"),
            "model_path": payload.get("model_path"),
        }
        return sampling_session_id

    def submit_sample(self, tenant: str, payload: dict) -> tuple[str, list[str]]:
        model_path = payload.get("model_path")
        if payload.get("sampling_session_id"):
            session = self.sampling_sessions[payload["sampling_session_id"]]
            if session["tenant"] != tenant:
                raise OwnershipError("sampling session does not belong to this tenant")
            model_path = model_path or session["model_path"]
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
        return future.request_id, sequence_ids

    async def _run_sample(
        self, request_id: str, payload: dict, lora_name: str | None, lora_path: str | None = None
    ) -> None:
        try:
            result = await self.backend.sample(payload, lora_name, lora_path)
            self.futures.resolve(request_id, {"op": "sample", **result})
        except asyncio.CancelledError:
            self.futures.fail(request_id, "cancelled", "user")
        except UserInputError as error:
            self.futures.fail(request_id, str(error), "user")
        except Exception as error:  # noqa: BLE001
            logger.exception("sample failed")
            self.futures.fail(request_id, f"{type(error).__name__}: {error}", "server")

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
        record = self.get_model(tenant, model_id)
        if kind != "sampler_weights":
            raise UserInputError(f"cannot sample from {model_path!r}: not a sampler_weights path")
        if not name.isdecimal() or int(name) not in record.published_sampler_versions:
            raise UserInputError(f"unknown sampler version {name} for {model_id}")
        return f"{model_id}@{name}", self._checkpoint_dir(model_id, "sampler_weights", name)

    async def sweep_leases(self) -> None:
        """Reclaim from stale tenants: cancel sampling, unload models, free
        slots. Training state dies with the lease; only checkpoints survive."""
        while True:
            await asyncio.sleep(30)
            await self._sweep_once()

    async def _sweep_once(self) -> None:
        now = time.monotonic()
        fresh_tenants = {
            session["tenant"]
            for session in self.sessions.values()
            if now - session["last_heartbeat"] < self.config.lease_timeout_s
        }

        def lease_expired(tenant: str) -> bool:
            # with no sessions at all there is no lease to expire
            return bool(self.sessions) and tenant not in fresh_tenants

        for request_id, (task, tenant) in list(self._sample_tasks.items()):
            if lease_expired(tenant):
                logger.warning(f"lease expired for tenant of sample {request_id}; cancelling")
                task.cancel()
        for model_id, record in list(self.models.items()):
            if not lease_expired(record.tenant):
                continue
            logger.warning(f"lease expired for {model_id}; freeing slot {record.slot}")
            stream = self.planner.stream(model_id)
            self.planner.remove_stream(model_id)
            del self.models[model_id]
            for request_id in stream.request_id_by_seq.values():
                future = self.futures.get(request_id, record.tenant)
                if future is not None and future.state == PENDING:
                    self.futures.fail(request_id, "lease expired", "user")
            async with self._backend_lock:
                await self.backend.unload_slot(record.slot)
            self.free_slots.add(record.slot)

    # -------- dispatch loop --------

    async def run(self) -> None:
        self._sweep_task = asyncio.create_task(self.sweep_leases())
        while True:
            item = self.planner.next_to_run()
            if item is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            async with self._backend_lock:
                if isinstance(item, BatchUnit):
                    await self._run_batch(item)
                else:
                    await self._run_barrier(item)

    async def _run_batch(self, batch: BatchUnit) -> None:
        # slot-contiguous order; outputs come back aligned to it
        refs = sorted(batch.datums, key=lambda ref: ref.stream.slot)
        slot_datums = [(ref.stream.slot, ref.datum) for ref in refs]
        self._batch_counter += 1
        run = self.backend.forward_backward if batch.op == CommandOp.FORWARD_BACKWARD else self.backend.forward_only
        try:
            outputs = await run(self._batch_counter, slot_datums, batch.loss_fn, batch.loss_fn_config)
        except UserInputError as error:
            await self._discard_windows(batch, str(error), "user")
            return
        except Exception as error:  # noqa: BLE001  infra failure: fail the affected windows, keep serving
            logger.exception(f"{batch.op} batch {self._batch_counter} failed")
            await self._discard_windows(batch, f"{type(error).__name__}: {error}", "server")
            return

        assert len(outputs) == len(refs), f"unit returned {len(outputs)} outputs for {len(refs)} datums"
        for ref, output in zip(refs, outputs, strict=True):
            request = ref.request
            if request.record_output(ref.local_index, output):
                self.futures.resolve(
                    request.command.request_id, {"op": request.command.op, "outputs": request.outputs}
                )
                ref.stream.finish(request)

    async def _discard_windows(self, batch: BatchUnit, error: str, category: str) -> None:
        """A failed batch poisons the gradient accumulation of every slot it
        touched, and that accumulation is shared with the other requests of the
        same window — so the whole open window of each affected stream fails
        and its slot's gradients are dropped."""
        streams = {ref.stream for ref in batch.datums}
        for stream in streams:
            for pending in list(stream.open_batch_run()):
                self.futures.fail(pending.command.request_id, error, category)
                stream.finish(pending)
        if batch.op == CommandOp.FORWARD_BACKWARD:
            for slot in sorted({stream.slot for stream in streams}):
                self._poisoned_slots[slot] = (error, category)
                await self.backend.zero_grads(slot)

    async def _run_barrier(self, barrier: BarrierUnit) -> None:
        try:
            results = await self._execute_barrier(barrier)
        except (UserInputError, OwnershipError) as error:
            self._fail_barrier(barrier, str(error), "user")
            return
        except Exception as error:  # noqa: BLE001
            logger.exception(f"{barrier.op} barrier failed")
            self._fail_barrier(barrier, f"{type(error).__name__}: {error}", "server")
            return
        if results is None:
            return
        for (stream, pending), result in zip(barrier.entries, results, strict=True):
            self.futures.resolve(pending.command.request_id, result)
            stream.finish(pending)

    def _fail_barrier(self, barrier: BarrierUnit, error: str, category: str) -> None:
        for stream, pending in barrier.entries:
            self.futures.fail(pending.command.request_id, error, category)
            stream.finish(pending)

    async def _execute_barrier(self, barrier: BarrierUnit) -> list[dict] | None:
        if barrier.op == CommandOp.OPTIM_STEP:
            await self._step_optimizers(barrier.entries)
            return None  # settled per slot
        ((stream, pending),) = barrier.entries  # every other barrier is single-entry
        record = self.models[stream.model_id]
        payload = pending.command.payload
        if barrier.op == CommandOp.SAVE_STATE:
            return await self._save_state(record, pending, payload)
        if barrier.op == CommandOp.LOAD_STATE:
            return await self._load_state(record, payload)
        if barrier.op == CommandOp.SAVE_WEIGHTS_FOR_SAMPLER:
            return await self._publish_sampler_version(record, payload)
        raise UserInputError(f"unknown barrier op {barrier.op!r}")

    async def _step_optimizers(self, entries: list) -> None:
        """Merged optim barriers settle per slot: one slot's failure must not
        mask another slot's completed step."""
        entries = [entry for entry in entries if not self._fail_if_poisoned(*entry)]
        if not entries:
            return
        outcomes = await self.backend.optim_step(
            {stream.slot: pending.command.payload["adam_params"] for stream, pending in entries}
        )
        for stream, pending in entries:
            outcome = outcomes[stream.slot]
            if "error" in outcome:
                self.futures.fail(pending.command.request_id, outcome["error"], "server")
            else:
                metrics = {key: float(value) for key, value in outcome.items()}
                self.futures.resolve(pending.command.request_id, {"op": "optim_step", "metrics": metrics})
            stream.finish(pending)

    def _fail_if_poisoned(self, stream, pending) -> bool:
        """Earlier batches of the window may have resolved before a later one
        failed and their gradients were discarded; the step must not silently
        run on the remainder."""
        poison = self._poisoned_slots.pop(stream.slot, None)
        if poison is None:
            return False
        error, category = poison
        self.futures.fail(
            pending.command.request_id,
            f"the gradient accumulation was discarded after a failed batch ({error}); resubmit the window",
            category,
        )
        stream.finish(pending)
        return True

    async def _save_state(self, record: ModelRecord, pending, payload: dict) -> list[dict]:
        name = payload["name"] or f"checkpoint-{pending.command.seq_id:06d}"
        _validate_checkpoint_segment(name)
        checkpoint_dir = self._checkpoint_dir(record.model_id, "weights", name)
        if not payload["overwrite"] and os.path.exists(checkpoint_dir):
            raise UserInputError(f"checkpoint {name!r} already exists; pass overwrite=True to replace it")
        await self.backend.save_slot(record.slot, checkpoint_dir)
        self._stamp_owner(checkpoint_dir, record.tenant)
        return [{"op": "save_state", "path": f"tinker://{record.model_id}/weights/{name}"}]

    async def _load_state(self, record: ModelRecord, payload: dict) -> list[dict]:
        source_id, kind, name = _parse_tinker_path(payload["path"])
        self._check_checkpoint_owner(self._checkpoint_dir(source_id, kind, name), record.tenant, payload["path"])
        await self.backend.load_slot(
            record.slot,
            record.lora_rank,
            record.lora_alpha,
            ckpt_path=self._checkpoint_dir(source_id, kind, name),
            load_optimizer=payload["optimizer"],
        )
        return [{"op": "load_state"}]

    async def _publish_sampler_version(self, record: ModelRecord) -> list[dict]:
        version = record.next_sampler_version
        record.next_sampler_version += 1
        path = self._checkpoint_dir(record.model_id, "sampler_weights", str(version))
        # disk is the commit point: the export makes the version exist; the
        # push only warms the engine cache
        await self.backend.export_slot(record.slot, record.lora_rank, record.lora_alpha, path)
        self._stamp_owner(path, record.tenant)
        record.published_sampler_versions.add(version)
        try:
            await self.backend.push_slot(
                record.slot, f"{record.model_id}@{version}", record.lora_rank, record.lora_alpha, lora_path=path
            )
        except Exception:  # noqa: BLE001
            logger.exception("adapter warm push failed; version %s will backfill from disk", version)
        return [
            {
                "op": "save_weights_for_sampler",
                "path": f"tinker://{record.model_id}/sampler_weights/{version}",
            }
        ]

    def _stamp_owner(self, checkpoint_dir: str, tenant: str) -> None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_dir) / "OWNER").write_text(tenant)

    def _check_checkpoint_owner(self, checkpoint_dir: str, tenant: str, shown_path: str) -> None:
        """Ownership travels with the checkpoint: it must outlive the source
        model's lease and gateway restarts."""
        owner_file = Path(checkpoint_dir) / "OWNER"
        if not owner_file.exists():
            raise UserInputError(f"unknown checkpoint {shown_path!r}")
        if owner_file.read_text() != tenant:
            raise OwnershipError(f"checkpoint {shown_path!r} does not belong to this tenant")

    def _checkpoint_dir(self, model_id: str, kind: str, name: str) -> str:
        root = os.path.realpath(self.config.checkpoint_root)
        path = os.path.realpath(f"{root}/{model_id}/{kind}/{name}")
        assert path.startswith(root + os.sep), f"checkpoint path {path!r} escapes {root!r}"
        return path


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
    """Client-provided segments become directory names under checkpoint_root;
    anything that could traverse out of it is rejected at the protocol edge."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", segment) is None:
        raise UserInputError(f"invalid checkpoint path segment {segment!r}")
