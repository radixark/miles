"""Custom fully-async multi-LoRA rollout function.

Mirrors ``examples/fully_async/fully_async_rollout.py`` (continuous background
producer + drain-a-batch) but for multi-LoRA:
  - ``generate`` sets ``rid = {adapter_name}_{uuid}`` next to ``lora_path`` for
    multi-LoRA samples, so the controller proxy can correlate/dummy by adapter,
  - sends through the multi-LoRA controller proxy (``--sglang-router-ip/port``
    pointed at the controller), which blocks retired adapters and returns a
    normal-shaped abort response for in-flight-retired stragglers,
  - recycles aborted/dummied groups back to the data source,
  - reuses ``generate_rollout_async``'s postprocess (dynamic filter,
    sample filter, logprob recompute).

The per-group logic is factored into ``process_group`` (testable without a
cluster); ``generate_fn`` defaults to the library ``generate_and_rm_group`` but
is injectable for tests.
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.sglang_rollout import (
    GenerateState,
    generate_and_rm_group,
    get_model_url,
)
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

GenerateFn = Callable[..., Any]


async def process_group(
    args, group: list[Sample], sampling_params: dict, generate_fn: GenerateFn, data_source
) -> list[Sample] | None:
    """Generate a group and either return it or recycle it.

    The rid is set inside ``generate`` (next to ``lora_path``) for multi-LoRA
    samples, so nothing here stamps it. Returns the completed group if it should
    be trained, or None if it was aborted/dummied (recycled back to
    ``data_source``).
    """
    result = await generate_fn(args, group, sampling_params)

    if any(s.status == Sample.Status.ABORTED for s in result):
        for s in result:
            s.reset_for_retry()
        try:
            data_source.add_samples([result])
        except Exception as e:
            logger.warning(f"Failed to recycle aborted group: {e}")
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
        result = await process_group(
            self.args, group, self.state.sampling_params, self.generate_fn, self.data_source
        )
        if result is not None:
            self.output_queue.put(result)

    def drain_completed(self) -> list[list[Sample]]:
        out = []
        while True:
            try:
                out.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def queue_size(self) -> int:
        return self.output_queue.qsize()


async def generate_rollout_multi_lora_async(
    args, rollout_id: int, data_source, generate_fn: GenerateFn = generate_and_rm_group
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Fully-async multi-LoRA rollout. Drain a batch from the background worker,
    then run the same postprocess as ``generate_rollout_async``."""
    assert args.rollout_global_dataset

    state = GenerateState(args)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None
    )
    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size

    worker = AsyncMultiLoRAWorker.get_or_create(args, data_source, generate_fn)

    # Read the active adapter set once — groups for deregistered adapters (still
    # in the queue from before the deregister) are stale and must not be trained
    # (their Megatron slot may have been cleaned up by reconcile). Discard them.
    from miles.ray.multi_lora_controller import get_multi_lora_controller
    active_names = set((await get_multi_lora_controller().active_adapters.remote()).keys())

    data: list[list[Sample]] = []
    start_time = time.time()
    last_progress = start_time
    while len(data) < target_data_size:
        made_progress = False
        for group in worker.drain_completed():
            adapter_name = group[0].adapter.name if group and group[0].adapter else None
            if adapter_name not in active_names:
                continue
            f = call_dynamic_filter(dynamic_filter, args, group)
            if not f.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=f.reason)
                continue
            if len(data) < target_data_size:
                data.append(group)
                made_progress = True

        if made_progress:
            last_progress = time.time()
        elif time.time() - last_progress > 30:
            logger.warning(f"No progress for 30s. queue={worker.queue_size()} collected={len(data)}/{target_data_size}")
            last_progress = time.time()

        if len(data) < target_data_size:
            await asyncio.sleep(0.01)

    data = sorted(data, key=lambda g: g[0].index)

    if (x := args.rollout_sample_filter_path) is not None:
        load_function(x)(args, data)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [s for g in data for s in g],
        url=get_model_url(args, "default"),
        sampling_params=state.sampling_params,
    )

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect())


def generate_rollout_multi_lora(args, rollout_id: int, data_source, evaluation: bool = False):
    if evaluation:
        raise ValueError("Evaluation not supported in multi-LoRA async rollout")
    return run(generate_rollout_multi_lora_async(args, rollout_id, data_source))
