"""The gateway service: sessions, models, streams, promises, and the single
dispatch loop.

Speaks only the internal language: server/ hands it decoded commands and
renders its results; runtime.py turns units into trainer batches. All backend
calls go through one loop / one lock: the trainer is an SPMD domain and must
see a single totally ordered unit stream.
"""

import asyncio
import logging
import time
import uuid

from miles.tinker.core.planner import BarrierUnit, Planner, WorkUnit
from miles.tinker.core.promise import PENDING, Promise, PromiseStore
from miles.tinker.core.stream import ModelStream
from miles.tinker.core.types import Command, GatewayConfig, ModelRecord, OwnershipError, UserInputError

logger = logging.getLogger(__name__)


class ExecutorBackend:
    """What runtime.py implements. Speaks rows and plain lists; core stays
    torch-free and miles-free."""

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        raise NotImplementedError

    async def unload_slot(self, slot: int) -> None:
        raise NotImplementedError

    async def forward_backward(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        """slot_rows: slot-sorted [(slot, row)]. Returns one
        {"loss": float, "logprobs": [float]} per row, in order."""
        raise NotImplementedError

    async def forward_only(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        raise NotImplementedError

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, float]:
        raise NotImplementedError

    async def save_slot(self, slot: int, path: str) -> None:
        raise NotImplementedError

    async def push_slot(self, slot: int, lora_name: str, rank: int, alpha: float) -> None:
        raise NotImplementedError

    async def sample(self, payload: dict, lora_name: str | None) -> dict:
        """-> {"sequences": [{"tokens", "logprobs", "stop_reason"}],
        "prompt_logprobs"?, "topk_prompt_logprobs"?}"""
        raise NotImplementedError


class TinkerService:
    def __init__(self, backend: ExecutorBackend, config: GatewayConfig) -> None:
        self.backend = backend
        self.config = config
        self.promises = PromiseStore()
        self.planner = Planner(config.unit_token_budget)
        self.models: dict[str, ModelRecord] = {}
        self.sessions: dict[str, dict] = {}
        self.sampling_sessions: dict[str, dict] = {}
        self.free_slots = set(range(config.n_slots))
        self._wake = asyncio.Event()
        self._backend_lock = asyncio.Lock()
        self._sample_tasks: dict[str, tuple] = {}  # request_id -> (task, tenant)
        self._create_tasks: set = set()
        self._arrival_counter = 0
        self._unit_counter = 0

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
        """Two-phase like every command: allocate now, initialize the slot behind the promise."""
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
        promise = self.promises.create(model_id, tenant)
        task = asyncio.create_task(self._run_create_model(promise.request_id, record))
        self._create_tasks.add(task)
        task.add_done_callback(self._create_tasks.discard)
        return promise.request_id, model_id

    async def _run_create_model(self, request_id: str, record: ModelRecord) -> None:
        try:
            async with self._backend_lock:
                await self.backend.load_slot(record.slot, record.lora_rank, record.lora_alpha)
        except Exception as error:
            self.models.pop(record.model_id, None)
            self.planner.remove_stream(record.model_id)
            self.free_slots.add(record.slot)
            self.promises.fail(request_id, str(error), "internal")
            return
        self.promises.resolve(request_id, {"kind": "create_model", "model_id": record.model_id})

    def get_model(self, tenant: str, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            raise UserInputError(f"unknown model {model_id!r}")
        if record.tenant != tenant:
            raise OwnershipError(f"model {model_id} does not belong to this tenant")
        return record

    # -------- command plane --------

    def submit(self, tenant: str, kind: str, payload: dict) -> str:
        """payload is server-decoded; content errors here are admission
        rejections and fail the promise (the SDK sees RequestFailedError)."""
        model_id = payload["model_id"]
        self.get_model(tenant, model_id)
        seq_id = payload["seq_id"]
        stream = self.planner.stream(model_id)

        # idempotency: the SDK resends the same seq_id after timeouts/410;
        # re-executing forward_backward would double-accumulate gradients
        if seq_id in stream.request_id_by_seq:
            return stream.request_id_by_seq[seq_id]

        promise = self.promises.create(model_id, tenant)
        stream.request_id_by_seq[seq_id] = promise.request_id
        try:
            self._admit(kind, payload)
        except UserInputError as error:
            self.promises.fail(promise.request_id, str(error), "user")
            # the rejected command still consumes its seq position, or the
            # stream would wait for it forever
            payload = {**payload, "rows": []}

        self._arrival_counter += 1
        stream.submit(
            Command(
                model_id=model_id,
                seq_id=seq_id,
                kind=kind,
                payload=payload,
                request_id=promise.request_id,
                arrival=self._arrival_counter,
            )
        )
        self._wake.set()
        return promise.request_id

    def _admit(self, kind: str, payload: dict) -> None:
        if kind not in ("forward_backward", "forward_only"):
            return
        rows = payload["rows"]
        if not rows:
            raise UserInputError("forward_backward with no data")
        if len(rows) > self.config.max_datums_per_request:
            raise UserInputError(
                f"{len(rows)} datums exceeds max_datums_per_request={self.config.max_datums_per_request}"
            )
        total_tokens = 0
        for index, row in enumerate(rows):
            if len(row["tokens"]) > self.config.max_tokens_per_datum:
                raise UserInputError(
                    f"datum {index}: {len(row['tokens'])} tokens exceeds {self.config.max_tokens_per_datum}"
                )
            total_tokens += len(row["tokens"])
        if total_tokens > self.config.max_tokens_per_request:
            raise UserInputError(
                f"{total_tokens} tokens exceeds max_tokens_per_request={self.config.max_tokens_per_request}"
            )

    def retrieve(self, tenant: str, request_id: str) -> Promise | None:
        """None -> the HTTP layer answers 410 and the SDK resubmits."""
        return self.promises.get(request_id, tenant)

    # -------- sampling plane (promise-based but never queues) --------

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
        lora_name = self._resolve_sampler(tenant, model_path) if model_path else None
        promise = self.promises.create(model_path or "base", tenant)
        sequence_ids = [f"seq-{uuid.uuid4().hex}" for _ in range(payload.get("num_samples", 1))]
        task = asyncio.create_task(self._run_sample(promise.request_id, payload, lora_name))
        self._sample_tasks[promise.request_id] = (task, tenant)
        task.add_done_callback(lambda _t, rid=promise.request_id: self._sample_tasks.pop(rid, None))
        return promise.request_id, sequence_ids

    async def _run_sample(self, request_id: str, payload: dict, lora_name: str | None) -> None:
        try:
            result = await self.backend.sample(payload, lora_name)
            self.promises.resolve(request_id, {"kind": "sample", **result})
        except asyncio.CancelledError:
            self.promises.fail(request_id, "cancelled", "user")
        except UserInputError as error:
            self.promises.fail(request_id, str(error), "user")
        except Exception as error:  # noqa: BLE001
            logger.exception("sample failed")
            self.promises.fail(request_id, f"{type(error).__name__}: {error}", "server")

    def cancel(self, tenant: str, request_id: str) -> None:
        """Cancel an in-flight sampling promise; training commands have no
        cancel in the protocol and are ignored."""
        if self.promises.get(request_id, tenant) is None:
            return
        entry = self._sample_tasks.get(request_id)
        if entry is not None:
            entry[0].cancel()

    def _resolve_sampler(self, tenant: str, model_path: str) -> str:
        model_id, kind, name = _parse_tinker_path(model_path)
        record = self.get_model(tenant, model_id)
        if kind != "sampler_weights":
            raise UserInputError(f"cannot sample from {model_path!r}: not a sampler_weights path")
        assert int(name) <= record.sampler_version, f"unknown sampler version {name} for {model_id}"
        return f"{model_id}@{name}"

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
        for request_id, (task, tenant) in list(self._sample_tasks.items()):
            if self.sessions and tenant not in fresh_tenants:
                logger.warning(f"lease expired for tenant of sample {request_id}; cancelling")
                task.cancel()
        for model_id, record in list(self.models.items()):
            if not self.sessions or record.tenant in fresh_tenants:
                continue
            logger.warning(f"lease expired for {model_id}; freeing slot {record.slot}")
            stream = self.planner.stream(model_id)
            self.planner.remove_stream(model_id)
            del self.models[model_id]
            for request_id in stream.request_id_by_seq.values():
                promise = self.promises.get(request_id, record.tenant)
                if promise is not None and promise.state == PENDING:
                    self.promises.fail(request_id, "lease expired", "user")
            async with self._backend_lock:
                await self.backend.unload_slot(record.slot)
            self.free_slots.add(record.slot)

    # -------- dispatch loop --------

    async def run(self) -> None:
        self._sweep_task = asyncio.create_task(self.sweep_leases())
        while True:
            unit = self.planner.next_unit()
            if unit is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            async with self._backend_lock:
                if isinstance(unit, WorkUnit):
                    await self._run_work(unit)
                else:
                    await self._run_barrier(unit)

    async def _run_work(self, unit: WorkUnit) -> None:
        # slot-contiguous order; outputs come back aligned to it
        refs = sorted(unit.rows, key=lambda ref: ref.stream.slot)
        slot_rows = [(ref.stream.slot, ref.row) for ref in refs]
        self._unit_counter += 1
        run = self.backend.forward_backward if unit.kind == "forward_backward" else self.backend.forward_only
        try:
            outputs = await run(self._unit_counter, slot_rows, unit.loss_fn, unit.loss_fn_config)
        except UserInputError as error:
            self._fail_riders(refs, str(error), "user")
            return
        except Exception as error:  # noqa: BLE001  infra failure: fail the riders, keep serving
            logger.exception(f"{unit.kind} unit {self._unit_counter} failed")
            self._fail_riders(refs, f"{type(error).__name__}: {error}", "server")
            return

        assert len(outputs) == len(refs), f"unit returned {len(outputs)} outputs for {len(refs)} rows"
        for ref, output in zip(refs, outputs, strict=True):
            request = ref.request
            request.outputs[ref.local_index] = output
            request.remaining -= 1
            if request.remaining == 0:
                self.promises.resolve(
                    request.command.request_id, {"kind": request.command.kind, "outputs": request.outputs}
                )
                ref.stream.finish(request)

    def _fail_riders(self, refs, error: str, category: str) -> None:
        seen: set[int] = set()
        for ref in refs:
            if id(ref.request) in seen:
                continue
            seen.add(id(ref.request))
            self.promises.fail(ref.request.command.request_id, error, category)
            ref.stream.finish(ref.request)

    async def _run_barrier(self, unit: BarrierUnit) -> None:
        try:
            results = await self._execute_barrier(unit)
        except (UserInputError, OwnershipError) as error:
            self._fail_barrier(unit, str(error), "user")
            return
        except Exception as error:  # noqa: BLE001
            logger.exception(f"{unit.kind} barrier failed")
            self._fail_barrier(unit, f"{type(error).__name__}: {error}", "server")
            return
        for (stream, pending), result in zip(unit.entries, results, strict=True):
            self.promises.resolve(pending.command.request_id, result)
            stream.finish(pending)

    def _fail_barrier(self, unit: BarrierUnit, error: str, category: str) -> None:
        for stream, pending in unit.entries:
            self.promises.fail(pending.command.request_id, error, category)
            stream.finish(pending)

    async def _execute_barrier(self, unit: BarrierUnit) -> list[dict]:
        if unit.kind == "optim_step":
            grad_norms = await self.backend.optim_step(
                {stream.slot: pending.command.payload["adam_params"] for stream, pending in unit.entries}
            )
            return [
                {"kind": "optim_step", "metrics": {"grad_norm": float(grad_norms[stream.slot])}}
                for stream, _ in unit.entries
            ]

        ((stream, pending),) = unit.entries
        record = self.models[stream.model_id]
        payload = pending.command.payload
        if unit.kind == "save_state":
            name = payload["name"] or f"checkpoint-{pending.command.seq_id:06d}"
            await self.backend.save_slot(record.slot, self._checkpoint_dir(record.model_id, "weights", name))
            return [{"kind": "save_state", "path": f"tinker://{record.model_id}/weights/{name}"}]
        if unit.kind == "load_state":
            source_id, kind, name = _parse_tinker_path(payload["path"])
            source = self.models.get(source_id)
            if source is None or source.tenant != record.tenant:
                raise OwnershipError(f"checkpoint {payload['path']} does not belong to this tenant")
            await self.backend.load_slot(
                record.slot,
                record.lora_rank,
                record.lora_alpha,
                ckpt_path=self._checkpoint_dir(source_id, kind, name),
                load_optimizer=payload["optimizer"],
            )
            return [{"kind": "load_state"}]
        if unit.kind == "save_weights_for_sampler":
            version = str(record.sampler_version + 1)
            path = self._checkpoint_dir(record.model_id, "sampler_weights", version)
            await self.backend.save_slot(record.slot, path)
            await self.backend.push_slot(
                record.slot, f"{record.model_id}@{version}", record.lora_rank, record.lora_alpha
            )
            record.sampler_version += 1  # every engine applied the push: the version is now sampleable
            return [
                {
                    "kind": "save_weights_for_sampler",
                    "path": f"tinker://{record.model_id}/sampler_weights/{version}",
                }
            ]
        raise UserInputError(f"unknown barrier kind {unit.kind!r}")

    def _checkpoint_dir(self, model_id: str, kind: str, name: str) -> str:
        return f"{self.config.checkpoint_root}/{model_id}/{kind}/{name}"


def _parse_tinker_path(path: str) -> tuple[str, str, str]:
    if not path.startswith("tinker://"):
        raise UserInputError(f"not a tinker path: {path!r}")
    parts = path.removeprefix("tinker://").split("/")
    if len(parts) != 3 or parts[1] not in ("weights", "sampler_weights"):
        raise UserInputError(f"malformed tinker path: {path!r}")
    return parts[0], parts[1], parts[2]
