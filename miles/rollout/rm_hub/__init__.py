import asyncio
import hashlib
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import aiohttp

from miles.utils.function_registry import load_function
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

from .deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl

_rm_timeout_executor: ThreadPoolExecutor | None = None
_rm_timeout_executor_lock = threading.Lock()


def _get_timeout_executor(max_workers: int) -> ThreadPoolExecutor:
    """Dedicated pool for graders run under --rm-timeout, so abandoned calls cannot exhaust the
    event loop's default executor. Created lazily; the size is fixed by the first caller."""
    global _rm_timeout_executor
    with _rm_timeout_executor_lock:
        if _rm_timeout_executor is None:
            _rm_timeout_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rm-timeout")
        return _rm_timeout_executor


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _resolve_reward_config(args, sample: Sample) -> tuple[str | None, str]:
    # Spec fields win when set; unset/empty fields fall back to sample metadata and process-wide args.
    spec = sample.reward_spec
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    custom_rm_path = (spec.custom_rm_path if spec is not None else None) or getattr(args, "custom_rm_path", None)
    rm_type = (
        (spec.rm_type if spec is not None else None) or metadata.get("rm_type") or getattr(args, "rm_type", None) or ""
    ).strip()
    return custom_rm_path, rm_type


def _timeout_reward(args) -> float | dict[str, float]:
    """Return {reward_key: 0.0} when args.reward_key is configured, otherwise 0.0."""
    reward_key = getattr(args, "reward_key", None)
    return {reward_key: 0.0} if reward_key else 0.0


async def async_rm(args, sample: Sample, **kwargs):
    """Score one sample, optionally limiting the wait with ``--rm-timeout``.

    A timeout logs a warning and returns ``{args.reward_key: 0.0}`` when a reward key is
    configured, otherwise ``0.0``. Built-in synchronous graders use a dedicated thread
    pool, created lazily with ``--rm-timeout-workers`` workers (default 8). Timed-out
    threads may keep running until the grader returns. Custom per-sample
    coroutines are awaited normally under the timeout, which cannot interrupt custom
    code that blocks synchronously. Batch-level custom reward functions are not covered.
    """
    timeout = getattr(args, "rm_timeout", None)
    if timeout is None:
        return await _async_rm(args, sample, **kwargs)
    try:
        return await asyncio.wait_for(_async_rm(args, sample, in_thread=True, **kwargs), timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "reward function exceeded --rm-timeout %ss for sample index=%s; assigning reward 0",
            timeout,
            sample.index,
        )
        return _timeout_reward(args)


async def _async_rm(args, sample: Sample, in_thread: bool = False, **kwargs):
    custom_rm_path, rm_type = _resolve_reward_config(args, sample)

    if custom_rm_path is not None:
        rm_function = load_function(custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    response = sample.response
    label = sample.label
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    if in_thread:
        executor = _get_timeout_executor(getattr(args, "rm_timeout_workers", 8))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor, copy_context().run, _rule_based_rm, rm_type, sample, response, label, metadata
        )
    return _rule_based_rm(rm_type, sample, response, label, metadata)


def _rule_based_rm(rm_type: str, sample: Sample, response, label, metadata):
    if rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "gemma_math":
        return get_gemma_math_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type == "deterministic_random":
        content = str(sample.tokens) + response
        content_hash = hashlib.sha256(content.encode()).digest()
        return int(content_hash[0]) % 2
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    inplace_set_reward_field: bool = False,
    **kwargs,
) -> list[int | float] | None:
    if inplace_set_reward_field:
        rewards = await batched_async_rm(args, samples, **kwargs)
        for sample, reward in zip(samples, rewards, strict=True):
            assert (
                sample.reward is None
            ), f"Overriding sample.reward from {sample.reward} to {reward}, is this intended?"
            sample.reward = reward
        return None

    if args.custom_rm_path is not None and not is_multi_lora_enabled(args):
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
