import asyncio
import copy
import inspect
import logging
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.utils.async_utils import run
from miles.utils.diffusion_data import Dataset as DiffusionDataset
from miles.utils.diffusion_rollout_response import apply_rollout_image_response
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.http_utils import post
from miles.utils.misc import SingletonMeta, load_function
from miles.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


def build_rollout_images_payload(
    args: Namespace,
    prompt: str,
    *,
    seed: int,
    num_outputs_per_prompt: int = 1,
    extra_sampling_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build JSON body for ``POST /rollout/images`` (``RolloutImageRequest``). Omits keys with ``None`` values
    except boolean return flags, which are always sent explicitly.
    """
    neg = getattr(args, "diffusion_negative_prompt", None)
    if neg is None:
        neg = [" "] * len(prompt)  # FlowGRPO default
    num_steps = args.diffusion_num_steps

    payload: dict[str, Any] = {
        "prompt": prompt,
        "seed": int(seed),
        "generator_device": getattr(args, "rollout_generator_device", "cuda"),
        "rollout_sde_type": getattr(args, "rollout_sde_type", "sde"),
        "rollout_noise_level": float(getattr(args, "diffusion_noise_level", 0.7)),
        "rollout_log_prob_no_const": bool(getattr(args, "rollout_log_prob_no_const", False)),
        "rollout_debug_mode": bool(getattr(args, "rollout_debug_mode", False)),
        "rollout_return_dit_env": True,
        "rollout_return_dit_trajectory": True,
        "num_outputs_per_prompt": num_outputs_per_prompt,
    }

    optional: dict[str, Any] = {
        "negative_prompt": neg,
        "width": getattr(args, "diffusion_width", None),
        "height": getattr(args, "diffusion_height", None),
        "num_inference_steps": num_steps,
        "guidance_scale": getattr(args, "diffusion_guidance_scale", None),
        "true_cfg_scale": getattr(args, "diffusion_true_cfg_scale", None),
    }

    for k, v in optional.items():
        if v is not None:
            payload[k] = v

    if extra_sampling_params:
        payload["extra_sampling_params"] = extra_sampling_params

    return payload


class GenerateState(metaclass=SingletonMeta):
    """Global state for sglang-diffusion image rollout."""

    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        # Merged into request ``extra_sampling_params``; may include per-call ``seed`` (stripped in ``generate``).
        self.extra_sampling_params: dict[str, Any] = {}
        sampling_seed_base = args.rollout_seed
        self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

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
                        extra_sampling_params=self.extra_sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate_microgroup(args: Namespace, microgroup: list[Sample], sampling_params: dict[str, Any]) -> list[Sample]:
    """Generate using traditional SGLang router with token-based workflow"""

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/rollout/images"

    # Prepare payload for sglang-diffusion server
    # SGL-D TODO: support seed list for multiple samples in one request
    # currently only support assigning the first seed, SGL-D generates samples with seed, seed+1, seed+2, ...
    payload = build_rollout_images_payload(args, microgroup[0].prompt, seed=sampling_params["rollout_first_seed"], num_outputs_per_prompt=len(microgroup))


    output = await post(url, payload)

    # Get diffusion response and log probs
    return [apply_rollout_image_response(sample, response) for sample, response in zip(microgroup, output)]


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
        if state.aborted:
            for sample in microgroup:
                sample.status = Sample.Status.ABORTED
            return microgroup

        with state.dp_rank_context() as _:
            if args.custom_generate_function_path is not None:
                custom_generate_func = load_function(args.custom_generate_function_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    microgroup = await custom_generate_func(args, microgroup, sampling_params, evaluation=evaluation)
                else:
                    microgroup = await custom_generate_func(args, microgroup, sampling_params)
            else:
                microgroup = await generate_microgroup(args, microgroup, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    tasks = []
    for idx in range(0, len(group), args.diffusion_microgroup_size):
        microgroup = group[idx:min(idx + args.diffusion_microgroup_size, len(group))]
        current_sampling_params = sampling_params.copy()
        current_sampling_params["rollout_first_seed"] = state.group_sampling_seeds[idx]
        tasks.append(
            asyncio.create_task(generate_and_rm_microgroup(args, microgroup, current_sampling_params, evaluation=evaluation))
        )

    microgroups = await asyncio.gather(*tasks)
    group = [sample for microgroup in microgroups for sample in microgroup]

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    # SGL-D TODO: support abort
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
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
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
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
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
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_config.path,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_config.input_key,
            label_key=dataset_config.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_config.metadata_key,
            tool_key=dataset_config.tool_key,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        # TODO: base sampling params

        # temperature=dataset_config.temperature,
        # top_p=dataset_config.top_p,
        # top_k=dataset_config.top_k,
        # max_new_tokens=dataset_config.max_response_len,
        # stop=args.rollout_stop,
        # stop_token_ids=args.rollout_stop_token_ids,
        # skip_special_tokens=args.rollout_skip_special_tokens,
        # no_stop_trim=True,
        # spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_config.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_config.inject_metadata(getattr(sample, "metadata", None))
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_config.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_config.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
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
