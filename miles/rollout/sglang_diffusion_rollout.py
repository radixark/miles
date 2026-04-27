import asyncio
import copy
import inspect
import logging
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.utils.async_utils import run
from miles.utils.diffusion_data import Dataset as DiffusionDataset
from miles.utils.diffusion_rollout_response import RolloutImageResponseParserActor
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.http_utils import post
from miles.utils.misc import SingletonMeta, load_function
from miles.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


def build_rollout_sampling_params(
    args: Namespace, 
    *, 
    extra_sampling_params: dict[str, Any] | None = None, 
    evaluation: bool = False
) -> dict[str, Any]:
    """Build static fields in JSON body for ``POST /rollout/generate`` (``RolloutImageRequest``). 
    """
    neg = getattr(args, "diffusion_negative_prompt", None)
    eval_steps = getattr(args, "diffusion_eval_num_steps", None)
    num_steps = int(eval_steps) if evaluation and eval_steps is not None else args.diffusion_num_steps

    sampling_params: dict[str, Any] = {
        "generator_device": getattr(args, "diffusion_generator_device", "cuda"),
        "negative_prompt": neg,
        "width": getattr(args, "diffusion_width", None),
        "height": getattr(args, "diffusion_height", None),
        "num_inference_steps": num_steps,
        "guidance_scale": getattr(args, "diffusion_guidance_scale", None),
        "true_cfg_scale": getattr(args, "diffusion_true_cfg_scale", None),
    }

    if evaluation:
        sampling_params["rollout"] = False
    else:
        sampling_params.update(
            {
                "rollout": True,
                "rollout_sde_type": getattr(args, "diffusion_sde_type", "sde"),
                "rollout_noise_level": float(getattr(args, "diffusion_noise_level", 0.7)),
                "rollout_log_prob_no_const": bool(getattr(args, "diffusion_log_prob_no_const", False)),
                "rollout_debug_mode": bool(getattr(args, "diffusion_debug_mode", False)),
                "rollout_return_denoising_env": True,
                "rollout_return_dit_trajectory": True,
            }
        )

    if extra_sampling_params:
        sampling_params["extra_sampling_params"] = extra_sampling_params

    return sampling_params

def build_rollout_generate_payload(
    sampling_params: dict[str, Any],
    prompt: str,
    *,
    num_outputs_per_prompt: int = 1,
) -> dict[str, Any]:
    """Build full JSON payload for ``POST /rollout/generate`` (``RolloutImageRequest``).
    """
    sampling_params["prompt"] = prompt
    if sampling_params["negative_prompt"] is None:
        sampling_params["negative_prompt"] = " "  # FlowGRPO default
    sampling_params["num_outputs_per_prompt"] = num_outputs_per_prompt
    return sampling_params

class GenerateState(metaclass=SingletonMeta):
    """Global state for sglang-diffusion image rollout."""

    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params = build_rollout_sampling_params(args)
        self.step_strategy_fn = (
            load_function(args.diffusion_step_strategy_path)
            if getattr(args, "diffusion_step_strategy_path", None)
            else None
        )
        sampling_seed_base = args.rollout_seed
        self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0
        self.node_id = ray.get_runtime_context().get_node_id()
        self.response_parser_actor = RolloutImageResponseParserActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=self.node_id, soft=False)
        ).remote()

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate_microgroup(
    args: Namespace, microgroup: list[Sample], sampling_params: dict[str, Any], *, evaluation: bool = False
) -> list[Sample]:
    """Generate using traditional SGLang router with token-based workflow"""

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/rollout/generate"

    # Prepare payload for sglang-diffusion server
    # SGL-D TODO: support seed list for multiple samples in one request
    # currently only support assigning the first seed, SGL-D generates samples with seed, seed+1, seed+2, ...
    if not evaluation and state.step_strategy_fn is not None:
        sde_indices, return_indices = state.step_strategy_fn(
            args,
            microgroup[0],
            int(sampling_params["num_inference_steps"]),
            int(sampling_params["seed"]),
        )
        sampling_params["rollout_sde_step_indices"] = sde_indices
        sampling_params["rollout_return_step_indices"] = return_indices
    else:
        sde_indices = None

    payload = build_rollout_generate_payload(
        sampling_params,
        microgroup[0].prompt,
        num_outputs_per_prompt=len(microgroup)
    )

    output = await post(url, payload)
    refs = [
        state.response_parser_actor.apply.remote(sample, response)
        for sample, response in zip(microgroup, output, strict=True)
    ]
    microgroup = await asyncio.to_thread(ray.get, refs)

    # Stash the SDE/training step indices on each sample so _train_core can
    # slice the full-length trajectory & rollout_log_probs down to the window.
    if sde_indices is not None:
        for sample in microgroup:
            md = sample.train_metadata or {}
            md["sde_step_indices"] = list(sde_indices)
            sample.train_metadata = md

    if not evaluation:
        # TODO: get real seeds from SGL-D
        for idx, sample in enumerate(microgroup):
            sample.seed = sampling_params["seed"] + idx

    return microgroup


async def generate_and_rm_microgroup(
    args: Namespace,
    microgroup: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    return_microgroup = []

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        with state.dp_rank_context() as _:
            if args.custom_generate_function_path is not None:
                custom_generate_func = load_function(args.custom_generate_function_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    microgroup = await custom_generate_func(args, microgroup, sampling_params, evaluation=evaluation)
                else:
                    microgroup = await custom_generate_func(args, microgroup, sampling_params)
            else:
                microgroup = await generate_microgroup(args, microgroup, sampling_params, evaluation=evaluation)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return microgroup

    # calculate the reward for the microgroup
    rewards = await batched_async_rm(args, microgroup)
    for sample, reward in zip(microgroup, rewards, strict=True):
        sample.reward = reward
    return microgroup

async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    tasks = []
    for idx in range(0, len(group), args.diffusion_microgroup_size):
        microgroup = group[idx:min(idx + args.diffusion_microgroup_size, len(group))]
        current_sampling_params = sampling_params.copy()
        current_sampling_params["seed"] = state.group_sampling_seeds[idx]
        tasks.append(
            asyncio.create_task(generate_and_rm_microgroup(args, microgroup, current_sampling_params, evaluation=evaluation))
        )

    microgroups = await asyncio.gather(*tasks)
    group = [sample for microgroup in microgroups for sample in microgroup]

    # for the rm that need the whole group, we will do the rm here
    if args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    # SGL-D TODO: support oversampling+filter & abort
    raise NotImplementedError("SGLang-Diffusion doesn't support abort")


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> RolloutFnTrainOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    # TODO: oversampling and abort
    assert args.over_sampling_batch_size == args.rollout_batch_size, "Now we don't support over sampling, please set --over_sampling_batch_size equal to --rollout_batch_size"

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample prompt: {[str(sample.prompt)]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout, prompt: {[str(sample.prompt)]}, reward: {sample.reward}",
    )

    # TODO: oversampling and abort
    # there are still some unfinished requests, abort them
    # aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect())


EVAL_PROMPT_DATASET = {}

# eval only
async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_config in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_config))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_config: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_config: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_config.cache_key + (args.hf_checkpoint,)
    if cache_key not in EVAL_PROMPT_DATASET:
        EVAL_PROMPT_DATASET[cache_key] = DiffusionDataset(
            path=dataset_config.path,
            prompt_key=dataset_config.input_key,
            metadata_key=dataset_config.metadata_key,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    base_sampling_params = build_rollout_sampling_params(args, evaluation=True)
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_config.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_config.inject_metadata(getattr(sample, "metadata", None))
            # Per-task dict so concurrent ``create_task`` calls never share one mutating mapping.
            # Train sets this inside ``generate_and_rm_group``; eval only needs ``rollout_microgroup_seed`` for images.
            sampling_params = base_sampling_params.copy()
            sampling_params["seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm_microgroup(
                        args,
                        [sample],
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_config.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        completed = await coro
        rows = completed if isinstance(completed, list) else [completed]
        if do_print:
            row = rows[0]
            logger.info(
                "eval_rollout_single_dataset example data, prompt: "
                f"{[str(row.prompt)]} "
                f"reward={row.reward}"
            )
            do_print = False
        data.extend(rows)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_config.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    # data_source.add_samples(aborted_samples)
    return output
