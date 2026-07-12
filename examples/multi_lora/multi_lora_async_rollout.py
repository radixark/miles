"""Fully-async multi-LoRA rollout: continuous background producer + collect-a-batch."""

import asyncio
import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from miles.ray.multi_lora_controller import AdaptersCache, get_multi_lora_controller
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group, get_model_url
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

GenerateFn = Callable[..., Any]

# Generate fns may return several samples per rollout; the manager flattens later.
Group = list[Sample | list[Sample]]


def iter_group_samples(group: Group):
    return itertools.chain.from_iterable(item if isinstance(item, list) else (item,) for item in group)


def first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


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


class AsyncMultiLoRAWorker:
    """Background producer: continuously generate groups into a thread-safe queue."""

    global_worker = None
    worker_lock = threading.Lock()

    def __init__(self, args, data_source, generate_fn: GenerateFn, concurrency: int = None) -> None:
        self.args = args
        self.data_source = data_source
        self.generate_fn = generate_fn
        self.concurrency = concurrency or args.rollout_batch_size
        self.running = True
        self.output_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

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
                    group = samples[0]
                    active.add(asyncio.create_task(self.process_and_enqueue(group)))

                await asyncio.sleep(0)
        finally:
            if active:
                await asyncio.wait(active)

    async def process_and_enqueue(self, group: list[Sample]) -> None:
        result = await process_group(self.args, group, self.state.sampling_params, self.generate_fn, self.data_source)
        if result is not None:
            self.output_queue.put(result)

    def queue_size(self) -> int:
        return self.output_queue.qsize()


async def generate_rollout_multi_lora_async(
    args, rollout_id: int, data_source, generate_fn: GenerateFn = generate_and_rm_group
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Fully-async multi-LoRA rollout. Collect a batch from the background worker,
    then run the same postprocess as ``generate_rollout_async``."""
    assert args.rollout_global_dataset

    state = GenerateState(args)

    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None
    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size

    worker = AsyncMultiLoRAWorker.get_or_create(args, data_source, generate_fn)

    # Groups whose submission-time slot version fell too far behind are dropped.
    max_staleness = getattr(args, "max_weight_staleness", None)

    data: list[Group] = []
    stale_dropped = 0
    staleness_values: list[int] = []
    start_time = time.time()
    last_progress = start_time
    queue_length = worker.queue_size()
    while len(data) < target_data_size:
        made_progress = False
        current_adapters = await AdaptersCache().get_all()
        # Pop one at a time so surplus groups stay queued for the next batch.
        while len(data) < target_data_size:
            try:
                group = worker.output_queue.get_nowait()
            except queue.Empty:
                break
            head = first_sample(group) if group else None
            adapter_name = head.adapter.name if head is not None and head.adapter else None
            if adapter_name not in current_adapters:
                continue  # adapter deregistered; drop
            if max_staleness is not None:
                stamped = head.metadata.get("slot_version")
                if stamped is not None:
                    staleness = current_adapters[adapter_name].version - stamped
                    if staleness > max_staleness:
                        for s in iter_group_samples(group):
                            s.reset_for_retry()
                        stale_dropped += 1
                        staleness_values.append(staleness)
                        logger.info(
                            f"Dropped stale group (adapter={adapter_name}, "
                            f"stamped={stamped}, current={current_adapters[adapter_name].version}, "
                            f"staleness={staleness} > max={max_staleness})"
                        )
                        continue
            f = call_dynamic_filter(dynamic_filter, args, group)
            if not f.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=f.reason)
                continue
            data.append(group)
            made_progress = True

        if made_progress:
            last_progress = time.time()
        elif time.time() - last_progress > 30:
            logger.warning(
                f"No progress for 30s. queue={worker.queue_size()} collected={len(data)}/{target_data_size}"
            )
            last_progress = time.time()

        if len(data) < target_data_size:
            await asyncio.sleep(0.01)

    if stale_dropped:
        logger.info(
            f"Staleness stats: dropped={stale_dropped}, "
            f"avg_staleness={sum(staleness_values) / len(staleness_values):.1f}, "
            f"max_staleness={max(staleness_values)}"
        )

    data = sorted(data, key=lambda g: first_sample(g).index)

    batch_adapters = sorted({first_sample(g).adapter.name for g in data if g and first_sample(g).adapter})
    if batch_adapters:
        await get_multi_lora_controller().record_batch_adapters.remote(rollout_id, batch_adapters)

    if (x := args.rollout_sample_filter_path) is not None:
        load_function(x)(args, data)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [s for g in data for s in iter_group_samples(g)],
        url=get_model_url(args, "default"),
        sampling_params=state.sampling_params,
    )

    metrics = {
        **metric_gatherer.collect(),
        "perf/fully_async/queue_length": queue_length,
        "perf/fully_async/batch_wait_time": time.time() - start_time,
        "perf/fully_async/stale_dropped": stale_dropped,
    }
    if staleness_values:
        metrics["perf/fully_async/stale_dropped_avg_staleness"] = sum(staleness_values) / len(staleness_values)

    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def generate_rollout_multi_lora(args, rollout_id: int, data_source, evaluation: bool = False):
    if evaluation:
        raise ValueError("Evaluation not supported in multi-LoRA async rollout")
    return run(generate_rollout_multi_lora_async(args, rollout_id, data_source))
