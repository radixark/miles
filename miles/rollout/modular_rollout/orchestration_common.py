import asyncio
import logging
from argparse import Namespace
from contextlib import contextmanager
from typing import Any

import numpy as np

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.compatibility import load_generate_function
from miles.rollout.modular_rollout.inference_wrapper import generate
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class GenerateState:
    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        if args.custom_generate_function_path is not None:
            self.generate_function = load_generate_function(args.custom_generate_function_path)
        else:
            self.generate_function = generate

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
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate_and_rm(
    state: GenerateState,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    args = state.args

    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            output = await state.generate_function(
                GenerateFnInput(
                    state=state,
                    sample=sample,
                    sampling_params=sampling_params,
                    evaluation=evaluation,
                )
            )
            sample = output.sample

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
    state: GenerateState, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    args = state.args

    if state.aborted:
        return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(state, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group
