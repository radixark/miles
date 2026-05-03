"""Fully-async rollout using random tokens and random rewards.

Exercises the entire async rollout <-> trainer loop for agentic-like toy data. 
Each Sample drives a multi-turn loop:

  1. start with a random ``input_ids`` sequence,
  2. POST it to the SGLang router's ``/generate`` endpoint
     (``ignore_eos=True``, ``max_new_tokens`` capped per turn),
  3. append the engine's response + random filler tokens to ``input_ids``
     and send the extended sequence as the next turn,
  4. stop when accumulated context exceeds ``MAX_CONTEXT_TOKENS``,
  5. assign a uniformly random reward.

Use this to smoke-test the async pipeline (rollout worker, weight sync,
trainer queue) and to stress-test SGLang under realistic multi-turn
long-context load. 

Wire it up via::

    --rollout-function-path random_async_rollout.generate_rollout_random_async
    --disable-rollout-global-dataset

The ``data_buffer`` argument is accepted for signature compatibility but
ignored; we generate ``Sample`` objects from scratch.
"""

import asyncio
import atexit
import logging
import os
import queue
import random
import threading
import time

from miles.rollout.data_source import DataSource
from miles.utils.async_utils import run
from miles.utils.http_utils import post as http_post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Random tokens never trigger EOS, so ``ignore_eos=True`` plus a hard
# ``max_new_tokens`` cap (drawn from this range) is what terminates each turn.
PROMPT_TOKEN_RANGE = (50, 2800)
MAX_TOKENS_RANGE = (512, 2048)
FILLER_RATIO = 3.0
# Cap on accumulated context per Sample. The multi-turn loop appends each
# turn's response + filler to the next request's input_ids until this is
# exceeded, then the Sample is finalised.
MAX_CONTEXT_TOKENS = 60000
# Concurrent in-flight SGLang requests target ``CONCURRENCY_PER_GPU *
# rollout_num_gpus``; each Sample drives one request at a time within its
# multi-turn loop, so this translates to a max-concurrent-Samples bound.
CONCURRENCY_PER_GPU = int(os.environ.get("RANDOM_ASYNC_CONCURRENCY_PER_GPU", "64"))
# Qwen3.5-35B vocab size; works for any model with vocab >= this.
VOCAB_SIZE = 151643
SGLANG_REQUEST_TIMEOUT_SECONDS = 10 * 60


def _rand_ids(n: int) -> list[int]:
    return [random.randrange(VOCAB_SIZE) for _ in range(n)]


async def _generate_one_random_sample(args, sample: Sample) -> Sample:
    """Multi-turn rollout against SGLang with random prompts, fillers, and reward.

    Send a request, take the engine's response, append response + random filler 
    to the running ``input_ids``, send the extended sequence as the next turn — 
    until the accumulated context exceeds ``MAX_CONTEXT_TOKENS``.
    """
    prompt_ids = _rand_ids(random.randint(*PROMPT_TOKEN_RANGE))
    current_ids = list(prompt_ids)
    accumulated_response: list[int] = []
    accumulated_log_probs: list[float] = []
    accumulated_loss_mask: list[int] = []
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    while True:
        payload = {
            "input_ids": current_ids,
            "sampling_params": {
                "max_new_tokens": random.randint(*MAX_TOKENS_RANGE),
                "temperature": 0.8,
                "ignore_eos": True,
            },
            "return_logprob": True,
        }
        max_new_tokens = payload["sampling_params"]["max_new_tokens"]
        try:
            data = await asyncio.wait_for(http_post(url, payload), timeout=SGLANG_REQUEST_TIMEOUT_SECONDS)
        except TimeoutError as e:
            raise TimeoutError(
                "SGLang /generate timed out after "
                f"{SGLANG_REQUEST_TIMEOUT_SECONDS}s "
                f"(sample_index={sample.index}, group_index={sample.group_index}, "
                f"input_tokens={len(current_ids)}, max_new_tokens={max_new_tokens})"
            ) from e
        meta = data["meta_info"]
        lp = meta["output_token_logprobs"]
        if not lp:
            raise RuntimeError(f"SGLang returned no output_token_logprobs: {data}")
        out_tokens = [item[1] for item in lp]
        out_log_probs = [item[0] for item in lp]

        filler = _rand_ids(int(len(out_tokens) * FILLER_RATIO))
        segment = out_tokens + filler
        accumulated_response.extend(segment)
        accumulated_log_probs.extend(out_log_probs + [0.0] * len(filler))
        accumulated_loss_mask.extend([1] * len(out_tokens) + [0] * len(filler))
        current_ids.extend(segment)
        sample.prefix_cache_info.add(meta)

        if "weight_version" in meta:
            sample.weight_versions.append(meta["weight_version"])

        if len(current_ids) > MAX_CONTEXT_TOKENS:
            break

    sample.tokens = current_ids
    sample.response_length = len(accumulated_response)
    sample.response = ""
    sample.reward = random.uniform(-1.0, 1.0)
    sample.loss_mask = accumulated_loss_mask
    sample.rollout_log_probs = accumulated_log_probs
    sample.status = Sample.Status.COMPLETED
    return sample


async def _generate_random_group(args, group: list[Sample]) -> list[Sample]:
    return list(await asyncio.gather(*[_generate_one_random_sample(args, s) for s in group]))


_global_worker: "AsyncRandomRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_worker(args) -> "AsyncRandomRolloutWorker":
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global random-async rollout worker...")
            _global_worker = AsyncRandomRolloutWorker(args)
            _global_worker.start()
        return _global_worker


def stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(stop_global_worker)


class AsyncRandomRolloutWorker:
    """Background asyncio loop that fills an output queue with random sample groups.

    Mirrors ``examples/fully_async/fully_async_rollout.AsyncRolloutWorker`` but
    skips the data buffer and reward model entirely.
    """

    def __init__(self, args):
        self.args = args
        self.running = True
        self.output_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.failure: BaseException | None = None
        self._sample_index = 0
        self._group_index = 0

    def _make_random_group(self) -> list[Sample]:
        group: list[Sample] = []
        for _ in range(self.args.n_samples_per_prompt):
            s = Sample()
            s.group_index = self._group_index
            s.index = self._sample_index
            self._sample_index += 1
            group.append(s)
        self._group_index += 1
        return group

    async def continuous_worker_loop(self):
        print("Continuous random-async rollout worker started")
        active: dict[asyncio.Task, int] = {}
        max_concurrent_groups = max(1, (CONCURRENCY_PER_GPU * self.args.rollout_num_gpus) // self.args.n_samples_per_prompt)
        gid_counter = 0

        try:
            while self.running:
                done = [task for task in active if task.done()]
                for task in done:
                    gid = active.pop(task)
                    result = task.result()
                    self.output_queue.put((gid, result))

                while len(active) < max_concurrent_groups and self.running:
                    group = self._make_random_group()
                    gid = gid_counter
                    gid_counter += 1
                    active[asyncio.create_task(_generate_random_group(self.args, group))] = gid
                    break

                await asyncio.sleep(1)
        finally:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            print("Continuous random-async rollout worker stopped")

    def worker_thread_func(self):
        try:
            asyncio.run(self.continuous_worker_loop())
        except Exception as e:
            self.failure = e
            self.running = False
            logger.exception("Random rollout worker crashed")
            raise

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started random-async worker thread")

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped random-async worker thread")

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        out: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                out.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


async def _generate_rollout_async(args, rollout_id: int) -> list[list[Sample]]:
    worker = get_global_worker(args)
    target = args.rollout_batch_size

    print(f"Random rollout {rollout_id}: collecting {target} groups, queue_depth={worker.get_queue_size()}")

    data: list[list[Sample]] = []
    start_time = time.time()
    last_progress = start_time
    progress_warn_interval = 30.0

    while len(data) < target:
        if worker.failure is not None:
            raise worker.failure
        if worker.worker_thread is not None and not worker.worker_thread.is_alive():
            raise RuntimeError("Random rollout worker exited without reporting a failure")

        for _, group in worker.get_completed_groups():
            if len(data) >= target:
                break
            data.append(group)
            last_progress = time.time()

        if time.time() - last_progress > progress_warn_interval:
            print(
                f"Random rollout {rollout_id}: no progress for {progress_warn_interval}s, "
                f"queue_depth={worker.get_queue_size()}, collected={len(data)}/{target}"
            )
            last_progress = time.time()

        await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Random rollout {rollout_id} done in {duration:.2f}s")
    data.sort(key=lambda g: g[0].index)
    return data


def generate_rollout_random_async(args, rollout_id, data_buffer: DataSource, evaluation: bool = False):
    """Entry point referenced by ``--rollout-function-path``.

    ``data_buffer`` is accepted for signature compatibility and ignored; the
    rollout fabricates its own random ``Sample`` groups.
    """
    if evaluation:
        raise ValueError("random_async_rollout does not support evaluation mode.")
    del data_buffer  # unused
    return run(_generate_rollout_async(args, rollout_id))
