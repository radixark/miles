import asyncio
import logging
import queue
import threading
from argparse import Namespace
from collections.abc import Callable

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.async_utils import get_async_loop
from miles.utils.misc import load_function
from miles.utils.types import Sample

from .sglang_rollout import GenerateState, abort, generate_and_rm_group

logger = logging.getLogger(__name__)


def _call_dynamic_filter(fn, *args, **kwargs) -> DynamicFilterOutput:
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


class InflightRolloutGenerator:
    """Continuously maintain a constant number of in-flight generation requests.

    This implements PipelineRL's actor loop behavior at the Miles rollout level:
    keep a fixed number of active request groups, pop finished ones, and
    immediately inject new prompts.
    """

    def __init__(self, args: Namespace, data_source: Callable[[int], list[list[Sample]]]) -> None:
        self.args = args
        self.data_source = data_source

        self._queue: queue.Queue[list[Sample]] = queue.Queue(maxsize=args.rollout_batch_size * 8)
        self._stop_event = threading.Event()
        self._started = False
        self._runner_ref = None

        self._dynamic_filter = (
            load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
        )

        # keep constant number of group tasks in-flight, each task expands into n_samples_per_prompt requests.
        self._target_inflight_groups = args.rollout_batch_size

        self._dropped_by_filter = 0

    def start(self) -> None:
        if self._started:
            return
        loop = get_async_loop().loop
        self._runner_ref = asyncio.run_coroutine_threadsafe(self._run_loop(), loop)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        try:
            get_async_loop().run(abort(self.args, rollout_id=-1))
        except Exception:
            pass
        try:
            GenerateState(self.args).reset()
        except Exception:
            pass
        if self._runner_ref is not None:
            try:
                self._runner_ref.result(timeout=10)
            except Exception:
                pass
        self._started = False

    def get_next_groups(self, num_groups: int) -> tuple[list[list[Sample]], dict[str, int]]:
        groups = []
        for _ in range(num_groups):
            groups.append(self._queue.get())
        metrics = {
            "rollout/dynamic_filter/dropped": self._dropped_by_filter,
        }
        self._dropped_by_filter = 0
        return groups, metrics

    async def _run_loop(self) -> None:
        state = GenerateState(self.args)
        pendings: set[asyncio.Task[list[Sample]]] = set()

        backoff_s = 0.01
        while not self._stop_event.is_set():
            while (
                not self._stop_event.is_set()
                and len(pendings) < self._target_inflight_groups
                and not self._queue.full()
            ):
                try:
                    samples = self.data_source(self.args.over_sampling_batch_size)
                except Exception as e:
                    logger.exception(f"PipelineRL inflight data_source failed: {e}")
                    await asyncio.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 1.0)
                    continue

                backoff_s = 0.01
                for group in samples:
                    pendings.add(
                        asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                    )

            if not pendings:
                await asyncio.sleep(0.01)
                continue

            done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    group = task.result()
                except Exception as e:
                    logger.exception(f"PipelineRL inflight generation task failed: {e}")
                    continue

                dynamic_filter_output = _call_dynamic_filter(self._dynamic_filter, self.args, group)
                if not dynamic_filter_output.keep:
                    self._dropped_by_filter += 1
                    continue

                while not self._stop_event.is_set():
                    try:
                        self._queue.put_nowait(group)
                        break
                    except queue.Full:
                        await asyncio.sleep(0.01)

        for task in pendings:
            task.cancel()
        if pendings:
            await asyncio.gather(*pendings, return_exceptions=True)
