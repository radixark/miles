"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
worker's output queue. Rollout production and training consumption run in parallel,
so per-iteration wall time moves from ``rollout_time + train_time`` toward
``max(rollout_time, train_time)``.

Requires the class-based rollout API (``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``)::

    --rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn

Evaluation requires a dedicated eval fleet (``--eval-num-gpus``); see
``miles/rollout/checkpoint_eval.py``.
"""

import asyncio
import logging
import time
from collections.abc import Iterator

import httpx

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainOutput,
)
from miles.rollout.checkpoint_eval import make_eval_generate_state
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.utils.http_utils import get
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

OUTPUT_QUEUE_MAX_GROUPS = 1000
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
    return min(versions) if versions else None


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via the router's /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query = 0.0

    async def get(self, args) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            data = await asyncio.wait_for(get(url), timeout=WEIGHT_VERSION_QUERY_TIMEOUT_SECS)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            # Transient router unavailability; the staleness filter is best-effort.
            logger.debug(f"Failed to query engine weight version: {e}")
            return self._value
        self._value = int(data["weight_version"])
        self._last_query = now
        return self._value


class FullyAsyncRolloutFn:
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Groups whose samples were aborted (e.g. by a
    weight update pausing generation) or whose weights are older than
    ``--max-weight-staleness`` are recycled back into the data source.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self._weight_version = _CachedWeightVersion()
        self._worker: asyncio.Task | None = None
        self._output: asyncio.Queue[Group] | None = None
        self._eval_state: GenerateState | None = None
        self._eval_prompt_dataset_cache: dict = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            self._output = asyncio.Queue(maxsize=OUTPUT_QUEUE_MAX_GROUPS)
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input.rollout_id)

    async def _call_eval(self, input: RolloutFnInput) -> RolloutFnOutput:
        if getattr(self.args, "eval_num_gpus", 0) <= 0:
            raise ValueError(
                "fully-async eval requires a dedicated eval fleet: set --eval-num-gpus > 0 "
                "(or run tools/checkpoint_eval_service.py against --save-hf checkpoints)"
            )
        if self._eval_state is None:
            self._eval_state = make_eval_generate_state(self.args)
        results = await run_eval_datasets(self._eval_state, self._eval_prompt_dataset_cache)
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task:
        [group] = self.data_source.get_samples(1)
        return asyncio.create_task(
            generate_and_rm_group(
                self.state,
                group,
                sampling_params=self.state.sampling_params.copy(),
                evaluation=False,
            )
        )

    async def _worker_loop(self):
        active: set[asyncio.Task] = set()
        while True:
            while len(active) < self._max_in_flight_groups():
                active.add(self._submit_one_group())
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                # Blocks when the queue is full: training lagging behind rollout
                # production pauses submission instead of growing the queue unboundedly.
                await self._output.put(task.result())

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> Group:
        queue_get = asyncio.create_task(self._output.get())
        while True:
            done, _ = await asyncio.wait(
                {queue_get, self._worker},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=NO_PROGRESS_WARN_SECS,
            )
            if queue_get in done:
                return queue_get.result()
            if self._worker in done:
                queue_get.cancel()
                self._worker.result()
                raise RuntimeError("fully-async rollout worker exited without an exception")
            logger.warning(
                f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s " f"(queued: {self._output.qsize()})"
            )

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        aborted_groups_recycled = 0
        stale_groups_recycled = 0
        staleness_values: list[int] = []
        do_print = True

        while len(data) < target_data_size:
            group = await self._next_group()
            assert len(group) == args.n_samples_per_prompt

            # A weight update paused generation mid-group: return it for re-sampling.
            if any(s.status == Sample.Status.ABORTED for s in _iter_samples(group)):
                self._recycle(group)
                aborted_groups_recycled += 1
                continue

            if args.max_weight_staleness is not None:
                oldest = group_oldest_weight_version(group)
                current = await self._weight_version.get(args)
                if oldest is not None and current is not None:
                    staleness = current - oldest
                    staleness_values.append(staleness)
                    if staleness > args.max_weight_staleness:
                        self._recycle(group)
                        stale_groups_recycled += 1
                        logger.info(
                            f"Recycled stale group (oldest_version={oldest}, current={current}, "
                            f"staleness={staleness} > max={args.max_weight_staleness})"
                        )
                        continue

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                    f"label: {sample.label}, reward: {sample.reward}"
                )
                do_print = False

            data.append(group)

        sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

        metrics = {
            "rollout/fully_async/queue_size": self._output.qsize(),
            "rollout/fully_async/aborted_groups_recycled": aborted_groups_recycled,
            "rollout/fully_async/stale_groups_recycled": stale_groups_recycled,
        }
        if staleness_values:
            metrics["rollout/fully_async/avg_staleness"] = sum(staleness_values) / len(staleness_values)
            metrics["rollout/fully_async/max_staleness"] = max(staleness_values)

        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, group: Group) -> None:
        for sample in _iter_samples(group):
            sample.reset_for_retry()
        self.data_source.add_samples([group])
