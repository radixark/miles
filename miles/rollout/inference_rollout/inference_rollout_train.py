import asyncio
import logging
from argparse import Namespace
from collections.abc import Callable

import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils.http_utils import get, post
from miles.utils.misc import as_completed_async, load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    args = state.args

    assert not state.aborted
    state.aborted = True

    urls = await get_worker_urls(args)
    logger.info(f"Abort request for {urls}")
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # make sure all the pending tasks are finished
    aborted_samples = []
    async for group in as_completed_async(pendings):
        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for sample in group:
            if sample.response and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)

    if args.partial_rollout:
        logger.info(f"Collected {sum(len(x) for x in aborted_samples)} partial samples into the data buffer")

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
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

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

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
