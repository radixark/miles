"""Multi-LoRA operation batching: one claim task per registration turns one
claimed client operation into one complete batch. The adapter selects whole
claimed batches with a persistent round-robin under a KIND LOCK — a selection
is all forward_backward or all forward, never mixed — and the BatchPlan,
shipped already converted as the output's conversion-metadata contribution, is
the only rollout-to-train control plane.

Nothing here generates: data operations arrive fully tokenized from the
client, and sampling happens against the router directly.
"""

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
    RolloutFnHandoff,
    RolloutFnInput,
    RolloutFnTrainOutput,
    RolloutPostprocessOptions,
)
from miles.rollout.multi_lora.operation_port import (
    BatchAbortPort,
    BatchResidencyPort,
    OperationQueuePort,
    RayMultiLoraBatchAbort,
    RayMultiLoraOperationQueue,
    RayTrainerResidencyPort,
)
from miles.utils.operation_contract import EmptyBatchTimeoutError
from miles.utils.types import AdapterRef, Sample

logger = logging.getLogger(__name__)


def _batch_plan_to_metadata(batch_plan: list[dict]) -> dict[str, Any]:
    """Distill one tinker selection's BatchPlan into conversion metadata.
    Selections are homogeneous: exactly one data-operation kind — mixed
    forward/forward_backward batches are structurally impossible, which is
    what keeps forward operations gradient-free without loss surgery.

    Correlation is batch-local (codex-rollout-fullparameter-design-0810 §3.3):
    each selected operation gets a small integer ``lane`` (its position in the
    selection), and the loss/result plane is keyed by lane — never by trainer
    slot, so operation identity survives any parameterization.

    The batch's ``BatchExecutionLease`` is the single binding truth (§5.3):
    it ships plain-encoded, and the conversion derives ``adapter_slots`` by
    joining ``operation_by_lane`` through it — the plan never stores a second
    copy of the binding."""
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
    }
    if kinds == {"forward"}:
        metadata["tinker_forward_only"] = True
    return metadata


def batch_plan_to_metadata(batch_plan: list[dict], lease) -> dict[str, Any]:
    """Build conversion metadata with the mandatory encoded batch lease."""
    metadata = _batch_plan_to_metadata(batch_plan)
    metadata["batch_execution_lease"] = lease_to_metadata(lease)
    return metadata


_CLAIM_POLL_S = 0.5

Tenant = tuple[str, str]

DATA_OPERATION_KINDS = ("forward_backward", "forward")


@dataclass(frozen=True)
class ClaimedOperationBatch:
    """One claimed client operation, decoded and stamped into a complete batch
    (external review 0813 §6.5): the single typed claim result that flows from
    the claim path through READY state and selection into the merge. The
    binding is the claim's fixed execution binding, resolved atomically with
    the claim (claim-and-bind) — the one dispatch truth; the long-lived
    runtime's AdapterRun view never is."""

    operation_id: str
    kind: str
    loss_spec: dict | None
    binding: Any  # duck-typed port binding; production ships ResidentBinding
    samples: list[list[Sample]]


def decode_operation(operation: dict, run: AdapterRun) -> ClaimedOperationBatch:
    """Decode one claimed operation into its stamped ClaimedOperationBatch:
    validate the data kind and payload, assign server-owned row indices, and
    stamp the registration's CURRENT serving identity (the version advances
    between batches; identity stays fixed) onto every sample."""
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
        # Row identity within the operation is server-owned: the result
        # plane returns per-datum logprobs in this order, and a negative
        # index is the DP-padding sentinel — a client-supplied value could
        # alias it (rows silently dropped) or collide in the collector.
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
    """The manager-level data source slot for tinker runs. Tinker has no
    dataset — every child pulls from the operation queue — so this only
    satisfies the manager's save/load/close surface."""

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
    """Operation-to-batch adapter (codex-rollout-fullparameter-design-0810
    §4.5): turns claimed client operations into whole training batches —
    persistent round-robin, homogeneous kind lock, coalesce timeout,
    registration fencing. Transports are injected ports (OperationQueuePort,
    BatchResidencyPort), so a future RolloutExecutor loads this adapter
    unchanged and unit tests need no Ray — "unchanged" is the executor/Ray
    boundary only. The adapter is NOT parameterization-neutral: its runtimes
    hold ``AdapterRun`` views and the claim path stamps samples with
    ``AdapterRef``, so a full-parameter deployment reuses the operation/
    result semantics but still needs a small sample-stamping extraction here
    (external review 0811: soften, do not pre-build the hook).

    The adapter never samples prompts, never generates, never scores, never
    builds Datums, and never touches residency policy — it only claims,
    selects, and converts."""

    def __init__(
        self,
        input: RolloutFnConstructorInput,
        operations: OperationQueuePort | None = None,
        residency: BatchResidencyPort | None = None,
        abort: BatchAbortPort | None = None,
    ):
        self.args = input.args
        self.operations = operations if operations is not None else RayMultiLoraOperationQueue()
        self.residency = residency if residency is not None else RayTrainerResidencyPort()
        self.abort = abort if abort is not None else RayMultiLoraBatchAbort()
        self.runtimes: dict[Tenant, AdapterRolloutRuntime] = {}
        self.rotation: deque[Tenant] = deque()
        self._ready = asyncio.Event()

    # ------------------------------ lifecycle ------------------------------

    async def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise ValueError(
                "MultiLoraOperationBatchFn does not serve eval; tinker runs have no server-side eval loop"
            )
        # READY streams only: a retiring registration's queued operations are
        # fenced terminal, so a child claim would never return for it.
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

    async def abort_handoff(self, handoff: RolloutFnHandoff, error: BaseException) -> None:
        """Finalize a leased selection when downstream batch preparation fails."""
        await self.abort.abort_batch(
            list(handoff.receipt["operation_ids"]),
            f"rollout batch preparation failed before trainer dispatch: {error}; the batch never "
            "reached the trainer and its gradient window is poisoned — resubmit the batch and "
            "optim_step again",
            handoff.receipt["lease"],
        )

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
        for runtime in self.runtimes.values():
            if runtime.state == AdapterRolloutRuntime.IDLE:
                runtime.state = AdapterRolloutRuntime.IN_FLIGHT
                runtime.task = asyncio.create_task(self._run_child(runtime))

    async def _claim_batch(self, runtime: AdapterRolloutRuntime) -> ClaimedOperationBatch:
        """Await the registration's next data-bearing operation and decode it
        into one complete stamped batch (0813 review §6.5). Blocking while the
        client queue is idle is normal: the runtime simply stays IN_FLIGHT and
        other adapters keep training. A malformed payload fails its own
        operation — never the adapter — and the claim loop continues."""
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
            runtime.state = AdapterRolloutRuntime.FAILED
        finally:
            self._ready.set()

    # ------------------------------ selection ------------------------------

    async def _select(self) -> list[AdapterRolloutRuntime]:
        """Collect READY child batches under the kind lock. The first selected
        operation locks the selection's kind (D11 homogeneity); other-kind
        READY batches stay READY for the next call. Two clocks: the empty-batch
        deadline before anything is selected, the coalesce window after."""
        soft_target = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        coalesce_wait = self.args.tinker_max_coalesce_wait_s
        empty_deadline = time.monotonic() + self.args.tinker_max_empty_wait_s
        selected: list[AdapterRolloutRuntime] = []
        kind_lock: str | None = None
        collected = 0
        coalesce_deadline: float | None = None

        try:
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
        except BaseException:
            # The selection has not acquired a lease yet. Cancellation or any
            # other coalescing failure returns this call's local picks to the
            # READY pool with their claimed outputs intact.
            for runtime in selected:
                if runtime.state == AdapterRolloutRuntime.SELECTED:
                    runtime.state = AdapterRolloutRuntime.READY
            raise

    def _pop_next_ready(self, kind_lock: str | None) -> AdapterRolloutRuntime | None:
        """Persistent round-robin over READY runtimes matching the kind lock:
        the cursor survives across selections so fast adapters cannot starve
        slow ones."""
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
        # Read-only pass: build the merged data and plan WITHOUT touching the
        # runtimes, so a failure anywhere up to and including lease
        # acquisition leaves every selected runtime READY with its output
        # intact (the claimed operation stays retryable at the next selection
        # instead of orphaning the only in-memory copy of an already-CLAIMED
        # output).
        operation_ids: list[str] = []
        lease_acquired = False
        lease = None
        lease_metadata = None
        try:
            for runtime in selected:
                claim = runtime.ready_output
                data.extend(claim.samples)
                # The claim's binding is the dispatch truth (resolved
                # atomically with the claim); the runtime's AdapterRun view
                # only names the metrics stream.
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
            # Validate and build everything that does not depend on residency
            # before acquiring the lease. A malformed selection therefore
            # remains retryable without minting a dispatch receipt.
            conversion_metadata = _batch_plan_to_metadata(batch_plan)
            operation_ids = [entry["operation_id"] for entry in batch_plan]
            # One immutable dispatch receipt for the whole selection: the
            # controller re-validates exact slot ownership before issuing it.
            lease = await self.residency.acquire_batch(
                [(entry["operation_id"], entry["binding"]) for entry in batch_plan]
            )
            lease_acquired = True
            lease_metadata = lease_to_metadata(lease)
            conversion_metadata["batch_execution_lease"] = lease_metadata
            output = RolloutFnTrainOutput(
                samples=data,
                metrics=metrics,
                # Converted HERE, not in the manager: the generic rollout plane
                # never recognizes tinker keys.
                conversion_metadata=conversion_metadata,
                # Whole client batches: zero-weight pads round the selection up
                # to the DP grid so the dynamic-GBS branch sizes the step to the
                # batch instead of trimming it.
                postprocess=RolloutPostprocessOptions(pad_to_dp=True),
                # Mint the lifecycle receipt where operation identity and the
                # lease are authoritative. Generic rollout infrastructure
                # forwards it opaquely and returns it to ``abort_handoff`` on
                # downstream failure.
                handoff=RolloutFnHandoff(
                    receipt={
                        "operation_ids": operation_ids,
                        "lease": lease_metadata,
                    }
                ),
            )
        except BaseException as error:
            if not lease_acquired:
                for runtime in selected:
                    runtime.state = AdapterRolloutRuntime.READY
            else:
                # Once a lease exists, retrying the same in-memory claim would
                # mint a second dispatch identity. Terminalize the exact batch
                # instead, preserving the original construction failure.
                abort_failed = False
                try:
                    await self.abort.abort_batch(
                        operation_ids,
                        f"failed to build rollout handoff after lease acquisition: {error}",
                        lease_metadata,
                    )
                except BaseException:  # cleanup must not mask the build error
                    abort_failed = True
                    logger.exception("failed to abort batch after rollout handoff construction error")
                if lease_metadata is None or abort_failed:
                    try:
                        await self.residency.release_batch(lease)
                    except BaseException:  # preserve the original build error
                        logger.exception("failed to release typed lease after rollout handoff construction error")
                for runtime in selected:
                    runtime.ready_output = None
                    runtime.state = AdapterRolloutRuntime.IDLE
            raise
        # The complete handoff now exists: only now consume the selected
        # outputs. A build failure above leaves them READY and retryable.
        for runtime in selected:
            runtime.ready_output = None
            runtime.state = AdapterRolloutRuntime.IDLE  # relaunches at the NEXT generate call
        return output
