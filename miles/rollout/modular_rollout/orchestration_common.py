import asyncio
import logging
from argparse import Namespace
from typing import Any

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub.single_turn import generate
from miles.rollout.modular_rollout.compatibility import load_generate_function
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

        self.generate_fn_semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = compute_sampling_params(
            args,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
        )

        self.generate_function = load_generate_function(args.custom_generate_function_path) or generate

        self.reset()

    def reset(self) -> None:
        self.aborted = False


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
    async with state.generate_fn_semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        output = await state.generate_function(
            GenerateFnInput(
                state=state,
                sample=sample,
                sampling_params=sampling_params,
                evaluation=evaluation,
            )
        )
        sample = output.samples

    # TODO change to `if not args.group_rm: do reward model` for more clarity after the refactor below
    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # TODO: unify the two branches into one if we decide to use list as output type
    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        await batched_async_rm(args, samples_need_reward, inplace_set_reward_field=True)
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
            current_sampling_params["sampling_seed"] = args.rollout_seed + idx
        tasks.append(
            asyncio.create_task(generate_and_rm(state, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)
    if state.aborted:
        return group

    if args.group_rm:
        await batched_async_rm(args, group, inplace_set_reward_field=True)

    return group


def compute_sampling_params(
    args,
    *,
    # after unifying configuration, this can be further refactored
    temperature,
    top_p,
    top_k,
    max_new_tokens,
):
    return dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )
