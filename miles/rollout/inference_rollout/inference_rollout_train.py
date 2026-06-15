import asyncio
import logging
from argparse import Namespace
from collections.abc import Callable

import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils import dumper_utils
from miles.utils.http_utils import get, post
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _consume_late_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning("[rollout-abort] pending task finished after abort with error: %r", exc)


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    args = state.args

    assert not state.aborted
    state.aborted = True
    logger.info(
        "[rollout-abort] start rollout_id=%s pending_tasks=%d partial_rollout=%s",
        rollout_id,
        len(pendings),
        args.partial_rollout,
    )
    timeout = float(getattr(args, "rollout_abort_timeout_sec", 30.0))

    logger.info("[rollout-abort] running custom abort handlers")
    try:
        await asyncio.wait_for(state.run_abort_handlers(pendings), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[rollout-abort] custom abort handlers timed out after %.1fs", timeout)
    logger.info("[rollout-abort] custom abort handlers finished")

    urls = await get_worker_urls(args)
    logger.info("[rollout-abort] sending sglang abort to %d workers: %s", len(urls), urls)
    try:
        sglang_results = await asyncio.wait_for(
            asyncio.gather(
                *[post(f"{url}/abort_request", {"abort_all": True}) for url in urls],
                return_exceptions=True,
            ),
            timeout=timeout,
        )
        sglang_failures = [result for result in sglang_results if isinstance(result, BaseException)]
        if sglang_failures:
            logger.warning("[rollout-abort] sglang abort had %d failures: %r", len(sglang_failures), sglang_failures)
    except asyncio.TimeoutError:
        logger.warning("[rollout-abort] sglang abort timed out after %.1fs", timeout)
    logger.info("[rollout-abort] sglang abort finished")

    aborted_samples = []
    completed_groups = 0
    pending_after_wait: set[asyncio.Task] = set()

    if pendings:
        if not args.partial_rollout:
            for task in pendings:
                task.cancel()
            done, pending_after_wait = await asyncio.wait(pendings, timeout=timeout)
            for task in done:
                _consume_late_task_result(task)
        else:
            done, pending_after_wait = await asyncio.wait(pendings, timeout=timeout)
            for task in done:
                try:
                    group = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    logger.warning("[rollout-abort] partial pending task failed after abort: %r", exc)
                    continue
                completed_groups += 1
                for sample in group:
                    if sample.response and "start_rollout_id" not in sample.metadata:
                        sample.metadata["start_rollout_id"] = rollout_id
                aborted_samples.append(group)
            for task in pending_after_wait:
                task.cancel()

        for task in pending_after_wait:
            task.add_done_callback(_consume_late_task_result)
        logger.info(
            "[rollout-abort] bounded pending cleanup done=%d still_pending=%d timeout=%.1fs collect_partial=%s",
            len(done),
            len(pending_after_wait),
            timeout,
            args.partial_rollout,
        )

    if args.partial_rollout:
        logger.info(f"Collected {sum(len(x) for x in aborted_samples)} partial samples into the data buffer")
    logger.info(
        "[rollout-abort] finish rollout_id=%s pending_completed=%d partial_groups_collected=%d",
        rollout_id,
        completed_groups,
        len(aborted_samples),
    )

    return aborted_samples


async def get_worker_urls(args: Namespace):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        return response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        return [worker["url"] for worker in response["workers"]]


def submit_generate_tasks(state: GenerateState, samples: list[list[Sample]]):
    return [
        asyncio.create_task(
            # submit a group of samples as a single task.
            generate_and_rm_group(
                state,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
        )
        for group in samples
    ]


async def generate_rollout_async(
    state: GenerateState, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    args = state.args
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    # instantiate data filters
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    pendings = set()
    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while len(data) + len(pendings) < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            pendings.update(submit_generate_tasks(state, samples))

        # wait for the generation to finish
        logger.debug(f"[rollout] Waiting on {len(pendings)} pending tasks, data={len(data)}/{target_data_size}")
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"[rollout] asyncio.wait returned: {len(done)} done, {len(pendings)} pending")
        for task in done:
            try:
                group: list[Sample] = task.result()
            except Exception as e:
                logger.error(f"[rollout] Task raised exception: {e!r}", exc_info=True)
                continue

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(state, pendings, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()

    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
    # There can be circumstances where users want to process all samples including filtered ones.
    if f := load_function(args.rollout_all_samples_process_path):
        f(args, all_samples, data_source)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [sample for group in data for sample in group],
        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        sampling_params=state.sampling_params,
    )

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
