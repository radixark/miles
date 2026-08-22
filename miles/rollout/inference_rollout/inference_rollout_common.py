import asyncio
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

from miles.rollout.base_types import (
    GenerateFnInput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.generate_hub.single_turn import generate
from miles.rollout.generate_utils.generate_endpoint_utils import policy_uses_routing_key
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class GenerateState:
    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
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

    # dashboard lifecycle probe: the semaphore wait is the queue; attempt_end fires before reward
    sink = None if evaluation else TrajectoryLifecycle().sink
    if sink is not None:
        sink.attempt_start(sample)

    # generate
    log_prefix = f"[sample={getattr(sample, 'index', '?')}]"
    logger.debug(f"{log_prefix} Waiting for semaphore...")
    try:
        async with state.generate_fn_semaphore:
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                return sample

            logger.debug(f"{log_prefix} Acquired semaphore, calling generate_function")
            if sink is not None:
                sink.gen_start(sample)
            # per-sample override, e.g. an eval dataset naming its own generate function
            generate_fn = load_generate_function(sample.generate_function_path) or state.generate_function
            output = await generate_fn(
                GenerateFnInput(
                    state=state,
                    sample=sample,
                    sampling_params=deepcopy(sampling_params),
                    evaluation=evaluation,
                )
            )
            sample = output.samples
            logger.debug(f"{log_prefix} generate_function returned")
    finally:
        if sink is not None:
            sink.attempt_end(sample)

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

    logger.debug(f"{log_prefix} generate_and_rm complete")
    return sample


async def generate_and_rm_group(
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    sample_done_callback: Callable[[], None] | None = None,
) -> list[Sample]:
    args = state.args

    if state.aborted:
        if sample_done_callback is not None:
            for _ in group:
                sample_done_callback()
        return group

    if policy_uses_routing_key(args):
        for sample in group:
            if sample.routing_key is None:
                sample.routing_key = str(uuid.uuid4())

    log_prefix = f"[group indices={[getattr(s, 'index', '?') for s in group]}]"
    logger.debug(f"{log_prefix} Starting group with {len(group)} samples")
    tasks: list[asyncio.Task[Sample | list[Sample]]] = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = args.rollout_seed + idx
        task = asyncio.create_task(generate_and_rm(state, sample, current_sampling_params, evaluation=evaluation))
        if sample_done_callback is not None:
            # fires on success, exception, and cancellation, so in-flight accounting is conserved
            task.add_done_callback(lambda _task: sample_done_callback())
        tasks.append(task)

    group_wait = asyncio.gather(*tasks)
    cancellation: asyncio.CancelledError | None = None
    group_error: BaseException | None = None
    while not group_wait.done():
        try:
            await asyncio.shield(group_wait)
        except asyncio.CancelledError as error:
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                if cancellation is None:
                    cancellation = error
                continue
            group_error = error
            break
        except BaseException as error:
            group_error = error
            break

    if group_error is None and group_wait.done():
        try:
            group_wait.result()
        except BaseException as error:
            group_error = error

    if group_error is not None:
        for task in tasks:
            if not task.done():
                task.cancel()

    terminal_wait = asyncio.gather(*tasks, return_exceptions=True)
    while not terminal_wait.done():
        try:
            await asyncio.shield(terminal_wait)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
    terminal_results = terminal_wait.result()
    terminal_errors = [result for result in terminal_results if isinstance(result, BaseException)]
    if cancellation is not None:
        cause = group_error
        if cause is None or isinstance(cause, asyncio.CancelledError):
            cause = next(
                (error for error in terminal_errors if not isinstance(error, asyncio.CancelledError)),
                cause,
            )
        if cause is None and terminal_errors:
            cause = terminal_errors[0]
        if cause is not None:
            raise cancellation from cause
        raise cancellation
    if group_error is not None:
        raise group_error
    if terminal_errors:
        raise terminal_errors[0]
    group = cast(list[Sample], [task.result() for task in tasks])
    logger.debug(f"{log_prefix} [group] All {len(group)} samples completed")
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


class InferenceRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self.eval_prompt_dataset_cache = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        from miles.rollout.inference_rollout.inference_rollout_train import generate_rollout_async

        output, aborted_samples = await generate_rollout_async(
            self.state, input.rollout_id, self.data_source.get_samples
        )
        self.data_source.add_samples(aborted_samples)
        return output

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        # Local: inference_rollout_eval imports GenerateState from this module.
        from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets

        state = input.generate_state or self.state
        results = await run_eval_datasets(state, self.eval_prompt_dataset_cache)
        return RolloutFnEvalOutput(data=results)
