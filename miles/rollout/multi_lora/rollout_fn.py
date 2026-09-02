import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from miles.ray.multi_lora.config import AdapterRun
from miles.ray.multi_lora.residency import lease_to_metadata
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnInput,
    RolloutFnTrainOutput,
    RolloutPostprocessOptions,
)
from miles.rollout.multi_lora.operation_port import (
    BatchResidencyPort,
    OperationQueuePort,
    RayMultiLoraOperationQueue,
    RayTrainerResidencyPort,
)
from miles.utils.operation_contract import EmptyBatchTimeoutError
from miles.utils.types import AdapterRef, Sample

logger = logging.getLogger(__name__)


def batch_plan_to_metadata(batch_plan: list[dict], lease) -> dict[str, Any]:
    kinds = {entry["operation_kind"] for entry in batch_plan}
    if len(kinds) != 1 or not kinds <= {"forward_backward", "forward"}:
        raise ValueError(f"tinker selection must be one homogeneous data kind, got {sorted(kinds)}")
    metadata: dict[str, Any] = {
        "batch_kind": "tinker",
        # Per-sample lanes in selection order (each entry's rows are contiguous).
        "tinker_operation_lanes": [
            lane for lane, entry in enumerate(batch_plan) for _ in range(entry["sample_count"])
        ],
        "tinker_loss_by_lane": {lane: entry.get("loss_spec") or {} for lane, entry in enumerate(batch_plan)},
        # The trainer completes these operations after the batch lands.
        "operation_by_lane": {lane: entry["operation_id"] for lane, entry in enumerate(batch_plan)},
        # Exact registration per lane: the batch commit dirties these streams,
        # never a trainer-reported name list.
        "registration_by_lane": {
            lane: (entry["name"], entry["registration_id"]) for lane, entry in enumerate(batch_plan)
        },
        # The lease is mandatory: a batch without its dispatch receipt is one
        # the trainer must reject, so the optional path may not exist here.
        "batch_execution_lease": lease_to_metadata(lease),
    }
    if kinds == {"forward"}:
        metadata["tinker_forward_only"] = True
    return metadata


_CLAIM_POLL_S = 0.5
# A FAILED child runtime returns to IDLE after this cooldown instead of starving its adapter until deregister.
_FAILED_RELAUNCH_COOLDOWN_S = 5.0

Tenant = tuple[str, str]

DATA_OPERATION_KINDS = ("forward_backward", "forward")


@dataclass(frozen=True)
class ClaimedOperationBatch:
    """One claimed operation as a complete batch; its binding (claim-and-bind) is the one dispatch truth."""

    operation_id: str
    kind: str
    loss_spec: dict | None
    binding: Any  # duck-typed port binding; production ships ResidentBinding
    samples: list[list[Sample]]


def decode_operation(operation: dict, run: AdapterRun) -> ClaimedOperationBatch:
    if operation["kind"] not in DATA_OPERATION_KINDS:
        raise ValueError(f"operation kind '{operation['kind']}' is not a data operation")
    payload = operation.get("payload") or {}
    raw_samples = payload.get("samples")
    if not raw_samples:
        raise ValueError(f"{operation['kind']} payload carries no samples")
    ref = AdapterRef(
        name=run.name,
        registration_id=run.registration_id,
        serving_version=run.version,
        slot=run.slot,
    )
    groups: list[list[Sample]] = []
    for i, raw in enumerate(raw_samples):
        raw = dict(raw)
        raw.setdefault("status", Sample.Status.COMPLETED.value)
        raw["index"] = i
        sample = Sample.from_dict(raw)
        sample.adapter = ref
        sample.metadata = {**run.config.metadata, **sample.metadata}
        groups.append([sample])
    return ClaimedOperationBatch(
        operation_id=operation["operation_id"],
        kind=operation["kind"],
        loss_spec=payload.get("loss"),
        binding=operation["binding"],
        samples=groups,
    )


class TinkerNullDataSource:
    """Dataset-less data source for tinker runs; only satisfies the manager's save/load/close surface."""

    dataset = ()

    def __init__(self, args):
        self.args = args

    def get_samples(self, num_samples: int):
        raise RuntimeError("tinker runs have no dataset; data arrives as client operations")

    def add_samples(self, samples) -> None:
        pass

    def save(self, rollout_id) -> None:
        pass

    def load(self, rollout_id=None) -> None:
        pass


class AdapterRolloutRuntime:
    """One per registration: at most one in-flight child claim task and one
    ready output."""

    IDLE = "IDLE"
    IN_FLIGHT = "IN_FLIGHT"
    READY = "READY"
    SELECTED = "SELECTED"
    FAILED = "FAILED"

    def __init__(self, run: AdapterRun):
        self.run = run
        self.state = self.IDLE
        self.ready_output: ClaimedOperationBatch | None = None
        self.task: asyncio.Task | None = None
        self.last_failure: float | None = None

    @property
    def ready_kind(self) -> str | None:
        if self.ready_output is None:
            return None
        return self.ready_output.kind

    def refresh(self, run: AdapterRun) -> None:
        """Serving version advances between batches; identity stays fixed."""
        self.run = run

    async def aclose(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown must not raise
                pass
        self.task = None


class MultiLoraOperationBatchFn:
    def __init__(
        self,
        input: RolloutFnConstructorInput,
        operations: OperationQueuePort | None = None,
        residency: BatchResidencyPort | None = None,
    ):
        self.args = input.args
        self.operations = operations if operations is not None else RayMultiLoraOperationQueue()
        self.residency = residency if residency is not None else RayTrainerResidencyPort()
        self.runtimes: dict[Tenant, AdapterRolloutRuntime] = {}
        self.rotation: deque[Tenant] = deque()
        self._ready = asyncio.Event()

    # ------------------------------ lifecycle ------------------------------

    async def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise ValueError(
                "MultiLoraOperationBatchFn does not serve eval; tinker runs have no server-side eval loop"
            )
        # READY streams only: a retiring registration's queued ops are fenced terminal, so a claim never returns.
        adapters = await self.operations.ready_streams()
        await self._reconcile(adapters)
        self._launch_idle_children()
        selected = await self._select()
        return await self._merge(selected)

    async def aclose(self) -> None:
        for runtime in list(self.runtimes.values()):
            await runtime.aclose()
        self.runtimes.clear()
        self.rotation.clear()

    # ------------------------------ runtimes ------------------------------

    async def _reconcile(self, adapters: dict[str, AdapterRun]) -> None:
        live = {(name, run.registration_id) for name, run in adapters.items()}
        for tenant in [t for t in self.runtimes if t not in live]:
            # Deregistered or re-registered: close the old tenant's runtime;
            # its late results are dropped with it (registration fencing).
            await self.runtimes.pop(tenant).aclose()
            logger.info(f"[tinker] closed child runtime for '{tenant[0]}' ({tenant[1][:8]})")
        for name, run in adapters.items():
            tenant = (name, run.registration_id)
            if tenant in self.runtimes:
                self.runtimes[tenant].refresh(run)
                continue
            self.runtimes[tenant] = AdapterRolloutRuntime(run)
            logger.info(f"[tinker] created child runtime for '{name}' ({run.registration_id[:8]})")
        self._sync_rotation()

    def _sync_rotation(self) -> None:
        in_queue = set()
        kept: deque[Tenant] = deque()
        while self.rotation:
            if (tenant := self.rotation.popleft()) in self.runtimes and tenant not in in_queue:
                kept.append(tenant)
                in_queue.add(tenant)
        for tenant in self.runtimes:
            if tenant not in in_queue:
                kept.append(tenant)
        self.rotation = kept

    def _launch_idle_children(self) -> None:
        now = time.monotonic()
        for runtime in self.runtimes.values():
            if runtime.state == AdapterRolloutRuntime.FAILED and (
                runtime.last_failure is None or now - runtime.last_failure >= _FAILED_RELAUNCH_COOLDOWN_S
            ):
                # FAILED is transient: after the cooldown the child relaunches instead of starving the adapter.
                runtime.state = AdapterRolloutRuntime.IDLE
            if runtime.state == AdapterRolloutRuntime.IDLE:
                runtime.state = AdapterRolloutRuntime.IN_FLIGHT
                runtime.task = asyncio.create_task(self._run_child(runtime))

    async def _claim_batch(self, runtime: AdapterRolloutRuntime) -> ClaimedOperationBatch:
        key = (runtime.run.name, runtime.run.registration_id)
        while True:
            operation = await self.operations.claim_data(key)
            if operation is None:
                await asyncio.sleep(_CLAIM_POLL_S)
                continue
            try:
                return decode_operation(operation, runtime.run)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - a bad payload fails its op, not the adapter
                logger.exception(f"[tinker] ({key[0]}) operation '{operation['operation_id']}' rejected: {e}")
                await self.operations.fail(operation["operation_id"], f"invalid operation payload: {e}", "user")

    async def _run_child(self, runtime: AdapterRolloutRuntime) -> None:
        try:
            output = await self._claim_batch(runtime)
            runtime.ready_output = output
            runtime.state = AdapterRolloutRuntime.READY
        except asyncio.CancelledError:
            runtime.state = AdapterRolloutRuntime.IDLE
            raise
        except Exception as e:
            # Child failure isolates to this adapter; other adapters keep going.
            logger.exception(f"[tinker] child for '{runtime.run.name}' failed: {e}")
            runtime.last_failure = time.monotonic()
            runtime.state = AdapterRolloutRuntime.FAILED
        finally:
            self._ready.set()

    # ------------------------------ selection ------------------------------

    async def _select(self) -> list[AdapterRolloutRuntime]:
        soft_target = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        coalesce_wait = self.args.tinker_max_coalesce_wait_s
        empty_deadline = time.monotonic() + self.args.tinker_max_empty_wait_s
        selected: list[AdapterRolloutRuntime] = []
        kind_lock: str | None = None
        collected = 0
        coalesce_deadline: float | None = None

        while True:
            runtime = self._pop_next_ready(kind_lock)
            if runtime is not None:
                selected.append(runtime)
                # Leave READY immediately or the round-robin would re-select
                # the same batch until the target is met (duplicated samples).
                runtime.state = AdapterRolloutRuntime.SELECTED
                kind_lock = runtime.ready_kind
                collected += sum(len(group) for group in runtime.ready_output.samples)
                if coalesce_deadline is None:
                    coalesce_deadline = time.monotonic() + coalesce_wait
                # Whole batches only: overshoot past the soft target is allowed,
                # trimming is not.
                if collected >= soft_target or len(selected) >= len(self.runtimes):
                    break
                continue

            now = time.monotonic()
            if selected:
                if now >= coalesce_deadline:
                    break
                timeout = coalesce_deadline - now
            else:
                if now >= empty_deadline:
                    raise EmptyBatchTimeoutError(
                        "no adapter produced a batch within "
                        f"--tinker-max-empty-wait-s ({self.args.tinker_max_empty_wait_s}s)"
                    )
                timeout = empty_deadline - now
            self._ready.clear()
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            except TimeoutError:
                continue
        return selected

    def _pop_next_ready(self, kind_lock: str | None) -> AdapterRolloutRuntime | None:
        for _ in range(len(self.rotation)):
            tenant = self.rotation.popleft()
            self.rotation.append(tenant)
            runtime = self.runtimes.get(tenant)
            if runtime is None or runtime.state != AdapterRolloutRuntime.READY:
                continue
            if kind_lock is not None and runtime.ready_kind != kind_lock:
                continue
            return runtime
        return None

    # ------------------------------ merge ------------------------------

    async def _merge(self, selected: list[AdapterRolloutRuntime]) -> RolloutFnTrainOutput:
        data: list[list[Sample]] = []
        batch_plan: list[dict] = []
        metrics: dict = {}
        try:
            for runtime in selected:
                claim = runtime.ready_output
                data.extend(claim.samples)
                name, registration_id = claim.binding.registration_key
                batch_plan.append(
                    dict(
                        name=name,
                        registration_id=registration_id,
                        operation_id=claim.operation_id,
                        operation_kind=claim.kind,
                        loss_spec=claim.loss_spec,
                        sample_count=sum(len(group) for group in claim.samples),
                        binding=claim.binding,
                    )
                )
                metrics[f"{runtime.run.name}/operation_samples"] = sum(len(group) for group in claim.samples)
            lease = await self.residency.acquire_batch(
                [(entry["operation_id"], entry["binding"]) for entry in batch_plan]
            )
        except BaseException:
            for runtime in selected:
                runtime.state = AdapterRolloutRuntime.READY
            raise
        # Acquisition succeeded: NOW consume the outputs.
        for runtime in selected:
            runtime.ready_output = None
            runtime.state = AdapterRolloutRuntime.IDLE  # relaunches at the NEXT generate call
        return RolloutFnTrainOutput(
            samples=data,
            metrics=metrics,
            conversion_metadata=batch_plan_to_metadata(batch_plan, lease),
            postprocess=RolloutPostprocessOptions(pad_to_dp=True),
        )
