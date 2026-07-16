"""Fully-async multi-LoRA rollout with per-adapter gradient accumulation.

A background producer generates continuously into per-adapter buffers. Each
train batch is collected by popping groups from the buffers round-robin, in
multiples of the adapter's ``min_groups_per_dp_split`` capped at its remaining
batch, so:

- any batch splits evenly across data-parallel ranks;
- an adapter's batch (``rollout_batch_size`` prompt groups, i.e.
  ``adapter_global_batch_size`` samples) is never overshot;
- adapters whose batch completes here are stamped as stepping.
"""

import asyncio
import itertools
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.ray.multi_lora_controller import AdaptersCache, get_multi_lora_controller
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group, get_model_url
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.multi_lora import EmptyBatchTimeoutError, min_groups_per_dp_split
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

GenerateFn = Callable[..., Any]

# Generate fns may return several samples per rollout; the manager flattens later.
Group = list[Sample | list[Sample]]


def iter_group_samples(group: Group):
    return itertools.chain.from_iterable(item if isinstance(item, list) else (item,) for item in group)


def first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def group_adapter_name(group: Group) -> str | None:
    head = first_sample(group) if group else None
    return head.adapter.name if head is not None and head.adapter else None


def group_sample_count(group: Group) -> int:
    return sum(1 for _ in iter_group_samples(group))


# Safety valve, same convention as fully_async's queue.Queue(maxsize=1000):
# never hit in practice, just bounds memory if training stalls entirely.
MAX_BUFFERED_GROUPS = 1000
EMPTY_BATCH_TIMEOUT_S = 30.0


class GroupBuffer:
    """One adapter's completed prompt groups: a FIFO queue you can also
    len(), and sweep for staleness. Bounded; the oldest group is dropped
    when a put exceeds the cap."""

    def __init__(self) -> None:
        self._groups: deque[Group] = deque(maxlen=MAX_BUFFERED_GROUPS)

    def __len__(self) -> int:
        return len(self._groups)

    def put(self, group: Group) -> None:
        self._groups.append(group)

    def get(self, n_groups: int) -> list[Group]:
        """Remove and return the n oldest groups (queue.Queue-style API)."""
        return [self._groups.popleft() for _ in range(n_groups)]

    def drop_stale(self, current_version: int, max_staleness: int | None) -> list[int]:
        """Drop groups generated too many weight versions ago; returns the
        staleness of each dropped group (for metrics)."""
        if max_staleness is None or not self._groups:
            return []
        kept: deque[Group] = deque(maxlen=MAX_BUFFERED_GROUPS)
        dropped: list[int] = []
        for group in self._groups:
            stamped = first_sample(group).metadata.get("slot_version")
            staleness = current_version - stamped if stamped is not None else 0
            if stamped is not None and staleness > max_staleness:
                for sample in iter_group_samples(group):
                    sample.reset_for_retry()
                dropped.append(staleness)
            else:
                kept.append(group)
        self._groups = kept
        return dropped


@dataclass
class TrainBatch:
    """One train batch: the groups for one train call, with its per-adapter bookkeeping."""

    groups: list[Group]
    group_counts: dict[str, int]  # prompt groups per adapter in this batch
    step_names: list[str]  # adapters whose adapter batch completes -> they step
    step_slots: list[int]


def remaining_groups(adapter) -> int:
    """Groups still needed to complete the adapter's batch."""
    remaining = adapter.config.rollout_batch_size - adapter.accumulated_groups
    assert remaining > 0, (
        f"adapter '{adapter.name}' accumulated_groups={adapter.accumulated_groups} >= "
        f"rollout_batch_size={adapter.config.rollout_batch_size}; batch accounting drifted"
    )
    return remaining


async def process_group(
    args, group: list[Sample], sampling_params: dict, generate_fn: GenerateFn, data_source
) -> Group | None:
    """Generate a group; returns None for aborted groups. The slot version is
    stamped at submission time (what the staleness filter compares against)."""
    adapter_name = group[0].adapter.name if group and group[0].adapter else None
    submission_version: int | None = None
    if adapter_name is not None:
        adapter = await AdaptersCache().get(adapter_name)
        submission_version = adapter.version if adapter is not None else None

    if submission_version is not None:
        for s in group:
            s.metadata["slot_version"] = submission_version

    result = await generate_fn(args, group, sampling_params)

    if submission_version is not None:
        for s in iter_group_samples(result):
            s.metadata["slot_version"] = submission_version

    if any(s.status == Sample.Status.ABORTED for s in iter_group_samples(result)):
        for s in iter_group_samples(result):
            s.reset_for_retry()
        # Re-queuing is not wired up (the per-adapter source is read-only).
        return None
    return result


class MultiLoRAWorkerMetrics:
    """The worker's cross-batch metric state, kept out of its buffer
    machinery: dynamic-filter drop counts, staleness drops, and per-adapter
    reward accumulation flushed as a step mean. Has its own lock — the
    producer thread records drops while the trainer thread flushes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dynamic_filter_drop_counts: dict[str, int] = defaultdict(int)
        self.stale_dropped = 0
        self.staleness_values: list[int] = []
        # Rewards of shipped samples, accumulated per adapter across train
        # batches and flushed as a step-mean metric when the adapter steps.
        self.reward_sums: dict[str, float] = defaultdict(float)
        self.reward_counts: dict[str, int] = defaultdict(int)

    def record_dynamic_filter_drop(self, reason: str) -> None:
        with self.lock:
            self.dynamic_filter_drop_counts[reason] += 1

    def record_stale_drops(self, staleness_values: list[int]) -> None:
        with self.lock:
            self.stale_dropped += len(staleness_values)
            self.staleness_values += staleness_values

    def record_shipped_rewards(self, args, data: list[Group], step_names: list[str]) -> dict[str, float]:
        """Accumulate the shipped batch's rewards per adapter; for adapters
        stepping with this batch, flush the mean over their whole adapter
        batch (accumulated across shipped batches, so it covers all
        ``adapter_global_batch_size`` samples of the step, not just this
        batch's slice).

        Counted at ship time, not train commit: a failed train call aborts the
        run anyway, so the distinction has no practical effect.
        """
        with self.lock:
            for group in data:
                name = group_adapter_name(group)
                if name is None:
                    continue
                for sample in iter_group_samples(group):
                    self.reward_sums[name] += sample.get_reward_value(args)
                    self.reward_counts[name] += 1

            metrics: dict[str, float] = {}
            for name in step_names:
                if (count := self.reward_counts.pop(name, 0)) > 0:
                    metrics[f"{name}/rollout/raw_reward/step_mean"] = self.reward_sums.pop(name) / count
                    metrics[f"{name}/rollout/raw_reward/step_n"] = count
            return metrics

    def discard_adapter(self, name: str) -> None:
        """Drop a retired adapter's partial reward accumulation."""
        with self.lock:
            self.reward_sums.pop(name, None)
            self.reward_counts.pop(name, None)

    def pop_metrics(self) -> dict[str, float]:
        with self.lock:
            metrics = {
                f"rollout/dynamic_filter/drop_{reason}": count
                for reason, count in self.dynamic_filter_drop_counts.items()
            }
            self.dynamic_filter_drop_counts.clear()
            metrics["perf/fully_async/stale_dropped"] = self.stale_dropped
            if self.staleness_values:
                metrics["perf/fully_async/stale_dropped_avg_staleness"] = sum(self.staleness_values) / len(
                    self.staleness_values
                )
                metrics["perf/fully_async/stale_dropped_max_staleness"] = max(self.staleness_values)
            self.stale_dropped = 0
            self.staleness_values = []
            return metrics


class AsyncMultiLoRAWorker:
    """Background producer filling bounded per-adapter completed-group buffers;
    the collection loop pops from them via ``get_groups``."""

    global_worker = None
    worker_lock = threading.Lock()

    def __init__(self, args, data_source, generate_fn: GenerateFn, concurrency: int = None) -> None:
        self.args = args
        self.data_source = data_source
        self.generate_fn = generate_fn
        self.concurrency = concurrency or args.rollout_batch_size
        self.running = True
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self.dynamic_filter = (
            load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None
        )
        # Guards the buffers: the producer thread puts completed groups while
        # get_groups (trainer side) pops them.
        self.buffer_lock = threading.Lock()
        self.buffers: dict[str, GroupBuffer] = defaultdict(GroupBuffer)
        # Fairness cursor: the adapter whose buffer get_groups visits first.
        # Advances past every visited adapter, persisting across calls and
        # batches, so adapters are served round-robin.
        self.rotation: deque[str] = deque()
        self.metrics = MultiLoRAWorkerMetrics()

    @classmethod
    def get_or_create(cls, args, data_source, generate_fn: GenerateFn, concurrency: int = None):
        with cls.worker_lock:
            if cls.global_worker is None or not cls.global_worker.worker_thread.is_alive():
                cls.global_worker = cls(args, data_source, generate_fn, concurrency)
                cls.global_worker.start()
        return cls.global_worker

    def start(self) -> None:
        self.worker_thread = threading.Thread(target=self.thread_main, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    @classmethod
    def stop_global(cls) -> None:
        with cls.worker_lock:
            if cls.global_worker is None:
                return
            cls.global_worker.stop()
            cls.global_worker = None

    def thread_main(self) -> None:
        asyncio.run(self.run_loop())

    async def run_loop(self) -> None:
        active: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        try:
            while self.running:
                done = {t for t in active if t.done()}
                for t in done:
                    try:
                        t.result()
                    except Exception as e:
                        logger.warning(f"generate task failed: {e}")
                    active.discard(t)

                while len(active) < max_concurrent and self.running:
                    samples = self.data_source.get_samples(1)
                    if not samples:
                        break
                    active.add(asyncio.create_task(self.process_and_enqueue(samples[0])))

                await asyncio.sleep(0.01)
        finally:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)

    async def process_and_enqueue(self, group: list[Sample]) -> None:
        result = await process_group(self.args, group, self.state.sampling_params, self.generate_fn, self.data_source)
        if result is None:
            return

        filter_result = call_dynamic_filter(self.dynamic_filter, self.args, result)
        if not filter_result.keep:
            if filter_result.reason:
                self.metrics.record_dynamic_filter_drop(filter_result.reason)
            return

        adapter_name = group_adapter_name(result)
        if adapter_name is None:
            return
        with self.buffer_lock:
            self.buffers[adapter_name].put(result)

    def queue_size(self) -> int:
        with self.buffer_lock:
            return sum(len(buffer) for buffer in self.buffers.values())

    def get_groups(
        self, snapshot: dict, num_samples: int, group_counts: dict[str, int]
    ) -> tuple[list[Group], dict[str, int]]:
        """Pop groups for the batch being collected. Returns the popped groups
        ([] when nothing is poppable right now) and an updated copy of
        ``group_counts`` (adapter name -> groups in the batch); passing the
        counts back on each fetch is what keeps a batch from overshooting an
        adapter's remaining groups.

        Pops round-robin from the cursor, one ``min_groups_per_dp_split`` at a
        time, until ``num_samples`` is covered (the final multiple may overshoot
        it) or no adapter can contribute. An adapter can't contribute when its
        buffer holds less than a whole multiple, or the batch already holds all
        its remaining groups.
        """
        adapters = {**snapshot["active"], **snapshot["retiring"]}
        dp_size = self.args.multi_lora_dp_size
        max_staleness = getattr(self.args, "max_weight_staleness", None)
        group_counts = dict(group_counts)  # updated copy; the argument is not modified
        popped: list[Group] = []
        popped_samples = 0

        with self.buffer_lock:
            # Adapters retired at the last reconcile sync point: their buffered
            # tail is discarded (base deregistration semantics), along with any
            # partially accumulated reward stats.
            for name in list(self.buffers):
                if name not in adapters:
                    self.buffers.pop(name)
                    self.metrics.discard_adapter(name)

            # Keep the rotation in sync with live adapters.
            self.rotation = deque(name for name in self.rotation if name in adapters)
            for name in sorted(set(adapters) - set(self.rotation)):
                self.rotation.append(name)

            while popped_samples < num_samples:
                made_progress = False
                for _ in range(len(self.rotation)):
                    name = self.rotation[0]
                    self.rotation.rotate(-1)
                    adapter = adapters[name]
                    buffer = self.buffers[name]
                    if dropped := buffer.drop_stale(adapter.version, max_staleness):
                        self.metrics.record_stale_drops(dropped)
                    min_groups_per_pop = min_groups_per_dp_split(adapter.config.n_samples_per_prompt, dp_size)
                    trainable_groups = len(buffer) // min_groups_per_pop * min_groups_per_pop
                    remaining_allowed_groups = max(0, remaining_groups(adapter) - group_counts.get(name, 0))
                    groups_to_pop = min(min_groups_per_pop, trainable_groups, remaining_allowed_groups)
                    if groups_to_pop <= 0:
                        continue
                    popped.extend(buffer.get(groups_to_pop))
                    popped_samples += groups_to_pop * adapter.config.n_samples_per_prompt
                    group_counts[name] = group_counts.get(name, 0) + groups_to_pop
                    made_progress = True
                    break
                if not made_progress:
                    break  # a full pass over rotation yielded nothing
        return popped, group_counts


async def collect_batch(args, worker: AsyncMultiLoRAWorker, snapshot: dict) -> TrainBatch:
    """Collect one train batch from the worker's buffers (same loop shape as
    fully_async's generate_rollout_async): keep popping group multiples until
    the batch reaches ``--global-batch-size`` samples, or it is non-empty and
    made no progress for ``--multi-lora-max-coalesce-wait-s`` (the target can
    be permanently unreachable when the live adapters' remaining batches are
    smaller than the target, so ship what there is).

    The remaining-groups math relies on the sequential trainer loop: the
    previous batch's ``mark_batch_trained`` has landed before this generate
    call, so the snapshot's ``accumulated_groups`` is current.
    """
    adapters = {**snapshot["active"], **snapshot["retiring"]}
    target_samples = args.global_batch_size
    wait_s = getattr(args, "multi_lora_max_coalesce_wait_s", 0.5)
    empty_wait_s = getattr(args, "multi_lora_max_empty_wait_s", EMPTY_BATCH_TIMEOUT_S)

    collected: list[Group] = []
    group_counts: dict[str, int] = {}
    total_samples = 0
    last_progress = time.time()
    last_warning = time.time()

    while total_samples < target_samples:
        groups, group_counts = worker.get_groups(snapshot, target_samples - total_samples, group_counts)
        if groups:
            collected.extend(groups)
            total_samples += sum(adapters[group_adapter_name(g)].config.n_samples_per_prompt for g in groups)
            last_progress = time.time()
            continue
        stalled_s = time.time() - last_progress
        if collected and stalled_s > wait_s:
            break
        if not collected and stalled_s > empty_wait_s:
            raise EmptyBatchTimeoutError(
                "No poppable groups collected before empty timeout; this likely means every live adapter is "
                "below min_groups_per_dp_split (or sources are exhausted). "
                f"queue={worker.queue_size()} active={sorted(snapshot['active'])} retiring={sorted(snapshot['retiring'])}"
            )
        if not collected and time.time() - last_warning > 30:
            logger.warning(
                "No completed groups for 30s. "
                f"queue={worker.queue_size()} active={sorted(snapshot['active'])} "
                f"retiring={sorted(snapshot['retiring'])}"
            )
            last_warning = time.time()
        await asyncio.sleep(0.01)

    step_names = sorted(name for name, count in group_counts.items() if count == remaining_groups(adapters[name]))
    return TrainBatch(
        groups=collected,
        group_counts=group_counts,
        step_names=step_names,
        step_slots=sorted(adapters[name].slot for name in step_names),
    )


async def generate_rollout_multi_lora_async(
    args, rollout_id: int, data_source, generate_fn: GenerateFn = generate_and_rm_group
) -> RolloutFnTrainOutput:
    """Collect one train batch and record its contents on the controller."""
    assert args.rollout_global_dataset

    state = GenerateState(args)
    worker = AsyncMultiLoRAWorker.get_or_create(args, data_source, generate_fn)
    start_time = time.time()
    queue_length = worker.queue_size()

    # Driver contract: generate is only called with live adapters, and the
    # sequential loop retires adapters and commits accumulated_groups only
    # between generate calls — so one snapshot serves the whole collection.
    snapshot = await get_multi_lora_controller().snapshot.remote()
    assert snapshot["active"] or snapshot["retiring"], "generate called with no live adapters"

    batch = await collect_batch(args, worker, snapshot)

    data = sorted(
        batch.groups,
        key=lambda group: (
            first_sample(group).adapter.slot if first_sample(group).adapter is not None else -1,
            first_sample(group).index,
        ),
    )

    # Per-sample adapter batch size (drives loss normalization) and batch-level step
    # decision (drives selective optimizer stepping), shipped via sample metadata.
    adapters = {**snapshot["active"], **snapshot["retiring"]}
    for group in data:
        adapter = adapters[group_adapter_name(group)]
        for sample in iter_group_samples(group):
            sample.metadata["adapter_global_batch_size"] = adapter.config.adapter_global_batch_size
    if data:
        head = first_sample(data[0])
        head.metadata["step_slots"] = list(batch.step_slots)
        head.metadata["step_adapter_names"] = list(batch.step_names)

    await get_multi_lora_controller().record_batch_adapters.remote(
        rollout_id, batch.group_counts, batch.step_names
    )

    if (x := args.rollout_sample_filter_path) is not None:
        load_function(x)(args, data)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [s for g in data for s in iter_group_samples(g)],
        url=get_model_url(args, "default"),
        sampling_params=state.sampling_params,
    )

    metrics = {
        **worker.metrics.pop_metrics(),
        **worker.metrics.record_shipped_rewards(args, data, batch.step_names),
        "perf/fully_async/queue_length": queue_length,
        "perf/fully_async/batch_wait_time": time.time() - start_time,
        "perf/fully_async/batch_adapters": len(batch.group_counts),
        "perf/fully_async/batch_prompt_groups": len(data),
        "perf/fully_async/batch_samples": sum(group_sample_count(group) for group in data),
        "perf/fully_async/batch_step_count": len(batch.step_names),
    }

    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def generate_rollout_multi_lora(args, rollout_id: int, data_source, evaluation: bool = False):
    if evaluation:
        raise ValueError("Evaluation not supported in multi-LoRA async rollout")
    return run(generate_rollout_multi_lora_async(args, rollout_id, data_source))
