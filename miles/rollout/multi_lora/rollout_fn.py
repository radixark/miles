"""Multi-LoRA rollout frontend: one child RolloutFn per registration, whole
child batches selected atomically by a persistent round-robin. The BatchPlan
(``RolloutFnTrainOutput.metadata``) is the only rollout-to-train control plane,
booking exactly one optimizer step per selected adapter."""

import asyncio
import copy
import inspect
import logging
import time
import uuid
from collections import deque

import ray

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.utils.adapter_config import AdapterRun
from miles.utils.misc import load_function
from miles.utils.multi_lora import EmptyBatchTimeoutError
from miles.utils.types import AdapterRef, RewardSpec, Sample

logger = logging.getLogger(__name__)

DEFAULT_CHILD_ROLLOUT_PATH = "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"

Tenant = tuple[str, str]


def leaf_sample_count(node) -> int:
    """Recursive leaf counter: multi-agent children may nest groups, so
    ``len(group)`` is not the sample count."""
    if isinstance(node, list):
        return sum(leaf_sample_count(child) for child in node)
    return 1


def first_sample(node):
    while isinstance(node, list):
        node = node[0]
    return node


def iter_leaf_samples(node):
    if isinstance(node, list):
        for child in node:
            yield from iter_leaf_samples(child)
    else:
        yield node


def leaf_rollout_count(node) -> int:
    """Distinct rollout executions among the leaves (samples without a
    ``rollout_id`` count as their own). The step normalizer divides by
    executions, matching the loss's per-rollout aggregation."""
    shared: set = set()
    solo = 0
    for sample in iter_leaf_samples(node):
        if sample.rollout_id is None:
            solo += 1
        else:
            shared.add(sample.rollout_id)
    return solo + len(shared)


class _AdapterDataSource:
    """Per-registration child data source: stamps serving identity and reward
    routing on every sample, so the child RolloutFn never learns about
    multi-LoRA, slots, or serving aliases."""

    def __init__(self, args, run: AdapterRun):
        config = run.config
        child_args = copy.copy(args)
        child_args.prompt_data = config.data
        child_args.input_key = config.input_key or args.input_key
        child_args.label_key = config.label_key or args.label_key
        child_args.metadata_key = config.metadata_key or args.metadata_key
        child_args.save = config.save or args.save
        child_args.load = config.save or args.load
        child_args.rollout_batch_size = config.rollout_batch_size
        child_args.n_samples_per_prompt = config.n_samples_per_prompt or args.n_samples_per_prompt
        child_args.start_rollout_id = 0
        # End-of-collection aborts cancel only this registration's requests.
        child_args.multi_lora_adapter_identity = (run.name, run.registration_id)
        self.args = child_args
        self.run = run
        # Buffer-capable: children recycle aborted/over-generated samples via
        # add_samples (a plain RolloutDataSource is read-only and raises).
        self.inner = RolloutDataSourceWithBuffer(child_args)

    @property
    def dataset(self):
        return self.inner.dataset

    def refresh(self, run: AdapterRun) -> None:
        """Serving version advances between batches; identity stays fixed."""
        self.run = run

    def _stamp(self, groups: list[list[Sample]]) -> list[list[Sample]]:
        run = self.run
        ref = AdapterRef(
            name=run.name,
            registration_id=run.registration_id,
            serving_version=run.version,
            slot=run.slot,
        )
        reward_spec = RewardSpec(rm_type=run.config.rm_type, custom_rm_path=run.config.custom_rm_path)
        for group in groups:
            for sample in group:
                sample.adapter = ref
                sample.reward_spec = reward_spec
                sample.metadata = {**run.config.metadata, **sample.metadata}
        return groups

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return self._stamp(self.inner.get_samples(num_samples))

    def add_samples(self, samples) -> None:
        self.inner.add_samples(samples)

    def save(self, rollout_id) -> None:
        self.inner.save(rollout_id)

    def load(self, rollout_id=None) -> None:
        self.inner.load(rollout_id)


class AdapterRolloutRuntime:
    """One per registration: at most one in-flight child call and one ready
    output."""

    IDLE = "IDLE"
    IN_FLIGHT = "IN_FLIGHT"
    READY = "READY"
    SELECTED = "SELECTED"
    FAILED = "FAILED"

    def __init__(self, args, run: AdapterRun):
        self.run = run
        self.data_source = _AdapterDataSource(args, run)
        path = run.config.rollout_function_path or DEFAULT_CHILD_ROLLOUT_PATH
        fn = load_function(path)
        child_input = RolloutFnConstructorInput(args=self.data_source.args, data_source=self.data_source)
        self.child_fn = fn(child_input) if inspect.isclass(fn) else fn
        self.state = self.IDLE
        self.ready_output: RolloutFnTrainOutput | None = None
        self.task: asyncio.Task | None = None
        self.error: BaseException | None = None

    @property
    def tenant(self) -> Tenant:
        return (self.run.name, self.run.registration_id)

    def refresh(self, run: AdapterRun) -> None:
        self.run = run
        self.data_source.refresh(run)

    async def aclose(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown must not raise
                pass
        self.task = None
        from miles.rollout.inference_rollout.compatibility import maybe_close

        await maybe_close(self.child_fn)


class MultiLoRARolloutFn:
    """Multi-LoRA wrapper: whole child batches only, persistent round-robin,
    soft target with overshoot, coalesce timeout, registration fencing."""

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.runtimes: dict[Tenant, AdapterRolloutRuntime] = {}
        self.rotation: deque[Tenant] = deque()
        self._ready = asyncio.Event()

    # ------------------------------ lifecycle ------------------------------

    async def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise ValueError(
                "MultiLoRARolloutFn does not serve eval; set --eval-function-path to "
                "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
            )
        adapters = await self._sampleable_adapters()
        await self._reconcile(adapters)
        self._launch_idle_children(input.rollout_id)
        slot_budget = await self._bindable_slot_count()
        selected = await self._select(slot_budget=slot_budget)
        return await self._merge(input.rollout_id, selected)

    async def aclose(self) -> None:
        for runtime in list(self.runtimes.values()):
            await runtime.aclose()
        self.runtimes.clear()
        self.rotation.clear()

    # ------------------------------ runtimes ------------------------------

    async def _sampleable_adapters(self) -> dict[str, AdapterRun]:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        snapshot = await asyncio.to_thread(ray.get, get_multi_lora_controller().snapshot.remote())
        return {**snapshot["active"], **snapshot["retiring"]}

    async def _reconcile(self, adapters: dict[str, AdapterRun]) -> None:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        live = {(name, run.registration_id) for name, run in adapters.items()}
        for tenant in [t for t in self.runtimes if t not in live]:
            # Deregistered or re-registered: close the old tenant's runtime;
            # its late results are dropped with it (registration fencing).
            await self.runtimes.pop(tenant).aclose()
            logger.info(f"[multilora] closed child runtime for '{tenant[0]}' ({tenant[1][:8]})")
        for name, run in adapters.items():
            tenant = (name, run.registration_id)
            if tenant in self.runtimes:
                self.runtimes[tenant].refresh(run)
                continue
            runtime = AdapterRolloutRuntime(self.args, run)
            self.runtimes[tenant] = runtime
            await asyncio.to_thread(
                ray.get,
                get_multi_lora_controller().resolve_num_step.remote(name, len(runtime.data_source.dataset)),
            )
            logger.info(f"[multilora] created child runtime for '{name}' ({run.registration_id[:8]})")
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

    def _launch_idle_children(self, rollout_id: int) -> None:
        for runtime in self.runtimes.values():
            if runtime.state == AdapterRolloutRuntime.IDLE:
                runtime.state = AdapterRolloutRuntime.IN_FLIGHT
                runtime.task = asyncio.create_task(self._run_child(runtime, rollout_id))

    async def _run_child(self, runtime: AdapterRolloutRuntime, rollout_id: int) -> None:
        try:
            output = await self._call_child(runtime.child_fn, RolloutFnTrainInput(rollout_id=rollout_id))
            if not isinstance(output, RolloutFnTrainOutput):
                output = RolloutFnTrainOutput(samples=output)
            if not output.samples:
                raise ValueError(f"child rollout for '{runtime.run.name}' returned an empty batch")
            self._validate_single_registration(runtime, output)
            runtime.ready_output = output
            runtime.state = AdapterRolloutRuntime.READY
        except asyncio.CancelledError:
            runtime.state = AdapterRolloutRuntime.IDLE
            raise
        except Exception as e:
            # Child failure isolates to this adapter; other adapters
            # keep rolling and training.
            logger.exception(f"[multilora] child rollout for '{runtime.run.name}' failed: {e}")
            runtime.error = e
            runtime.state = AdapterRolloutRuntime.FAILED
        finally:
            self._ready.set()

    @staticmethod
    async def _call_child(fn, input: RolloutFnTrainInput):
        is_async = inspect.iscoroutinefunction(fn)
        if not is_async and callable(fn):
            # Class-based child rollout fns implement an async __call__.
            is_async = inspect.iscoroutinefunction(fn.__call__)
        if is_async:
            return await fn(input)
        # Sync child: run off the event loop so other adapters keep generating.
        output = await asyncio.to_thread(fn, input)
        if inspect.iscoroutine(output):
            output = await output
        return output

    def _validate_single_registration(self, runtime: AdapterRolloutRuntime, output: RolloutFnTrainOutput) -> None:
        expected = runtime.tenant
        for group in output.samples:
            ref = first_sample(group).adapter
            if ref is None or (ref.name, ref.registration_id) != expected:
                raise ValueError(
                    f"child rollout for '{expected[0]}' produced samples for "
                    f"{None if ref is None else (ref.name, ref.registration_id)}; "
                    "children must draw from their own adapter data source"
                )

    # ------------------------------ selection ------------------------------

    async def _bindable_slot_count(self) -> int:
        """Advisory slot budget: the controller's plan_bind at merge is
        the authoritative admission; this only stops the selection from picking
        far more adapters than can possibly bind."""
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        budget = await asyncio.to_thread(ray.get, get_multi_lora_controller().bindable_slot_count.remote())
        return max(1, budget)

    async def _select(self, slot_budget: int = 1_000_000) -> list[AdapterRolloutRuntime]:
        soft_target = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        coalesce_wait = self.args.multi_lora_max_coalesce_wait_s
        empty_deadline = time.monotonic() + self.args.multi_lora_max_empty_wait_s
        selected: list[AdapterRolloutRuntime] = []
        collected = 0
        coalesce_deadline: float | None = None

        while True:
            runtime = self._pop_next_ready()
            if runtime is not None:
                selected.append(runtime)
                # Leave READY immediately or the round-robin would re-select
                # the same batch until the target is met (duplicated samples).
                runtime.state = AdapterRolloutRuntime.SELECTED
                collected += leaf_sample_count(runtime.ready_output.samples)
                if coalesce_deadline is None:
                    coalesce_deadline = time.monotonic() + coalesce_wait
                # Whole batches only: overshoot past the soft target is allowed,
                # trimming is not. The slot budget caps how many adapters can
                # bind for one train call.
                if collected >= soft_target or len(selected) >= slot_budget:
                    break
                continue

            now = time.monotonic()
            # Two separate clocks (review P1-2): before the first ready batch we
            # wait on the empty-batch timeout; once something is selected we
            # only coalesce until the deadline.
            if selected:
                if now >= coalesce_deadline:
                    break
                timeout = coalesce_deadline - now
            else:
                if now >= empty_deadline:
                    raise EmptyBatchTimeoutError(
                        "no adapter produced a complete batch within "
                        f"--multi-lora-max-empty-wait-s ({self.args.multi_lora_max_empty_wait_s}s)"
                    )
                timeout = empty_deadline - now
            self._ready.clear()
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            except TimeoutError:
                continue
        return selected

    def _pop_next_ready(self) -> AdapterRolloutRuntime | None:
        """Persistent round-robin over READY runtimes: the cursor survives
        across selections so fast adapters cannot starve slow ones."""
        for _ in range(len(self.rotation)):
            tenant = self.rotation.popleft()
            self.rotation.append(tenant)
            runtime = self.runtimes.get(tenant)
            if runtime is not None and runtime.state == AdapterRolloutRuntime.READY:
                return runtime
        return None

    # ------------------------------ merge ------------------------------

    async def _merge(self, rollout_id: int, selected: list[AdapterRolloutRuntime]) -> RolloutFnTrainOutput:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        # Authoritative admission: the controller reserves a slot per
        # tenant — keep-warm hit, free, or LRU-evictable — while the selection
        # is still inside generate. Tenants that no longer fit re-queue READY.
        txn_id = uuid.uuid4().hex
        plan_by_tenant = await asyncio.to_thread(
            ray.get,
            get_multi_lora_controller().plan_bind.remote(txn_id, [r.tenant for r in selected]),
        )
        admitted = [r for r in selected if r.tenant in plan_by_tenant]
        for runtime in selected:
            if runtime.tenant not in plan_by_tenant:
                runtime.state = AdapterRolloutRuntime.READY
        if not admitted:
            await asyncio.to_thread(ray.get, get_multi_lora_controller().abort_bind.remote(txn_id))
            raise EmptyBatchTimeoutError("selection admitted no adapters: no bindable slots")
        selected = admitted

        data: list[list[Sample]] = []
        batch_plan: list[dict] = []
        metrics: dict = {}
        for runtime in selected:
            output = runtime.ready_output
            runtime.ready_output = None
            runtime.state = AdapterRolloutRuntime.IDLE  # relaunches at the NEXT generate call (publish gate)
            run = runtime.run
            data.extend(output.samples)
            plan_entry = plan_by_tenant[runtime.tenant]
            batch_plan.append(
                dict(
                    name=run.name,
                    registration_id=run.registration_id,
                    bound_slot=plan_entry["slot"],
                    evict=plan_entry["evict"],
                    actual_sample_count=leaf_sample_count(output.samples),
                    actual_rollout_count=leaf_rollout_count(output.samples),
                    prompt_group_sizes=[leaf_sample_count([group]) for group in output.samples],
                )
            )
            for key, value in (output.metrics or {}).items():
                metrics[f"{run.name}/{key}"] = value
            # Per-run reward observability: one line per selected batch.
            rewards = []
            for group in output.samples:
                for sample in group:
                    try:
                        value = sample.get_reward_value(self.args)
                    except (AttributeError, KeyError, TypeError):
                        continue
                    if isinstance(value, (int, float)):
                        rewards.append(float(value))
            if rewards:
                reward_mean = sum(rewards) / len(rewards)
                metrics[f"{run.name}/rollout_reward_mean"] = reward_mean
                logger.info(f"[multilora] ({run.name}) selected batch: n={len(rewards)} reward_mean={reward_mean:.4f}")

        step_names = sorted(entry["name"] for entry in batch_plan)
        await asyncio.to_thread(
            ray.get,
            get_multi_lora_controller().record_train_selection.remote(rollout_id, step_names),
        )

        return RolloutFnTrainOutput(
            samples=data,
            metrics=metrics,
            metadata={"batch_plan": batch_plan, "train_txn_id": txn_id},
        )


def _iter_leaves(node):
    if isinstance(node, list):
        for child in node:
            yield from _iter_leaves(child)
    else:
        yield node
