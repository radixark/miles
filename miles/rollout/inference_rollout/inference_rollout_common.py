import asyncio
import contextvars
import logging
from argparse import Namespace
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

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
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class AbortHandle:
    key: str
    abort: Callable[[], Awaitable[None]]
    label: str = ""


_CURRENT_ABORT_TASK: contextvars.ContextVar[asyncio.Task | None] = contextvars.ContextVar(
    "miles_current_abort_task",
    default=None,
)


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

        self._abort_handles: dict[str, AbortHandle] = {}
        self._abort_handles_by_task: dict[asyncio.Task, set[str]] = {}
        self._abort_handlers_lock = asyncio.Lock()
        self.reset()

    def reset(self) -> None:
        self.aborted = False
        self._abort_handles.clear()
        self._abort_handles_by_task.clear()

    def bind_current_abort_task(self, task: asyncio.Task | None):
        return _CURRENT_ABORT_TASK.set(task)

    def reset_current_abort_task(self, token) -> None:
        _CURRENT_ABORT_TASK.reset(token)

    async def add_abort_handle(self, handle: AbortHandle) -> None:
        bound_task = _CURRENT_ABORT_TASK.get()
        async with self._abort_handlers_lock:
            self._abort_handles[handle.key] = handle
            if bound_task is not None:
                self._abort_handles_by_task.setdefault(bound_task, set()).add(handle.key)
            logger.info(
                "[rollout-abort] registered abort handle key=%s label=%s bound_task=%s active_handles=%d",
                handle.key,
                handle.label,
                id(bound_task) if bound_task is not None else None,
                len(self._abort_handles),
            )

    async def remove_abort_handle(self, key: str) -> None:
        async with self._abort_handlers_lock:
            removed = self._abort_handles.pop(key, None) is not None
            empty_tasks = []
            for task, keys in self._abort_handles_by_task.items():
                keys.discard(key)
                if not keys:
                    empty_tasks.append(task)
            for task in empty_tasks:
                self._abort_handles_by_task.pop(task, None)
            logger.info(
                "[rollout-abort] removed abort handle key=%s active_handles=%d removed=%s",
                key,
                len(self._abort_handles),
                removed,
            )

    async def add_abort_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        key = f"handler:{id(handler)}"
        await self.add_abort_handle(AbortHandle(key=key, abort=handler, label=getattr(handler, "__name__", repr(handler))))

    async def remove_abort_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        await self.remove_abort_handle(f"handler:{id(handler)}")

    async def run_abort_handlers(self, tasks: set[asyncio.Task] | None = None) -> None:
        async with self._abort_handlers_lock:
            if tasks is None:
                keys = set(self._abort_handles)
            else:
                keys = set()
                for task in tasks:
                    keys.update(self._abort_handles_by_task.get(task, set()))
            handles = [self._abort_handles.pop(key) for key in keys if key in self._abort_handles]
            for key in keys:
                empty_tasks = []
                for task, task_keys in self._abort_handles_by_task.items():
                    task_keys.discard(key)
                    if not task_keys:
                        empty_tasks.append(task)
                for task in empty_tasks:
                    self._abort_handles_by_task.pop(task, None)
        logger.info("[rollout-abort] running %d abort handles", len(handles))
        if not handles:
            return
        results = await asyncio.gather(*(handle.abort() for handle in handles), return_exceptions=True)
        success_count = sum(1 for result in results if not isinstance(result, BaseException))
        logger.info("[rollout-abort] abort handles finished success=%d failed=%d", success_count, len(results) - success_count)
        for handle, result in zip(handles, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning("Abort handle failed key=%s label=%s error=%r", handle.key, handle.label, result)


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
    log_prefix = f"[sample={getattr(sample, 'index', '?')}]"
    logger.debug(f"{log_prefix} Waiting for semaphore...")
    async with state.generate_fn_semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        logger.debug(f"{log_prefix} Acquired semaphore, calling generate_function")
        output = await state.generate_function(
            GenerateFnInput(
                state=state,
                sample=sample,
                sampling_params=deepcopy(sampling_params),
                evaluation=evaluation,
            )
        )
        sample = output.samples
        logger.debug(f"{log_prefix} generate_function returned")

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
    state: GenerateState, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    args = state.args

    if state.aborted:
        return group

    log_prefix = f"[group indices={[getattr(s, 'index', '?') for s in group]}]"
    logger.debug(f"{log_prefix} Starting group with {len(group)} samples")
    tasks = []
    token = state.bind_current_abort_task(asyncio.current_task())
    try:
        for idx, sample in enumerate(group):
            current_sampling_params = sampling_params.copy()
            if getattr(args, "sglang_enable_deterministic_inference", False):
                current_sampling_params["sampling_seed"] = args.rollout_seed + idx
            tasks.append(
                asyncio.create_task(generate_and_rm(state, sample, current_sampling_params, evaluation=evaluation))
            )
    finally:
        state.reset_current_abort_task(token)

    group = await asyncio.gather(*tasks)
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
        from miles.rollout.inference_rollout.inference_rollout_eval import eval_rollout_single_dataset

        assert not self.state.args.group_rm, "Group RM is not supported for eval rollout"

        coros = []
        for dataset_cfg in getattr(self.state.args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(self.state, dataset_cfg, self.eval_prompt_dataset_cache))
        results_list = await asyncio.gather(*coros)
        results = {k: v for r in results_list for k, v in r.items()}
        return RolloutFnEvalOutput(data=results)
