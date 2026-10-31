import asyncio
import hashlib
import random

import aiohttp

from miles.utils.args.custom_function import CustomFunctionConfig, _compute_config_class
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl

# `math`, `deepscaler` and `gemma_math` already grade the last `\boxed{}` themselves (deepscaler and
# gemma_math only after their end-of-reasoning marker), so their `boxed_` variants grade the response
# unchanged. `dapo` reads `Answer: ...`
# instead, so the extracted answer is re-wrapped in that form.
_BOXED_GRADES_ORIGINAL_RESPONSE = {"math", "deepscaler", "gemma_math"}


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


def _read_row_custom_rm_path(sample: Sample) -> str | None:
    """`metadata['custom_rm_path']`, or None when unset or empty."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    path = metadata.get("custom_rm_path")
    if path is not None and not isinstance(path, str):
        raise TypeError(f"metadata['custom_rm_path'] must be a function path string, got {type(path).__name__}")
    return path or None


def _run_custom_rm_path(args) -> str | None:
    run_hook = args.custom_rm_path
    return run_hook.path if isinstance(run_hook, CustomFunctionConfig) else run_hook


def _resolve_row_custom_rm_path(args, path: str) -> CustomFunctionConfig:
    """A row naming the run's hook reuses its config; a row cannot supply config for any other hook."""
    if path == _run_custom_rm_path(args) and isinstance(args.custom_rm_path, CustomFunctionConfig):
        return args.custom_rm_path
    if _compute_config_class(load_function(path), path=path) is not None:
        raise ValueError(
            f"metadata['custom_rm_path']={path!r} names a reward function with its own config, which a dataset row "
            "cannot carry; set it as --custom-rm-path instead"
        )
    return CustomFunctionConfig(path=path)


def _resolve_reward_config(args, sample: Sample) -> tuple[CustomFunctionConfig | None, str]:
    # Spec fields win when set; unset/empty fields fall back to sample metadata and process-wide args.
    spec = sample.reward_spec
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    row_custom_rm_path = _read_row_custom_rm_path(sample)
    if spec is not None and spec.custom_rm_path:
        custom_rm_path = spec.custom_rm_path
    elif row_custom_rm_path is not None:
        custom_rm_path = _resolve_row_custom_rm_path(args, row_custom_rm_path)
    else:
        custom_rm_path = args.custom_rm_path
    if isinstance(custom_rm_path, str):
        custom_rm_path = CustomFunctionConfig(path=custom_rm_path)
    rm_type = ((spec.rm_type if spec is not None else None) or metadata.get("rm_type") or args.rm_type or "").strip()
    return custom_rm_path, rm_type


async def async_rm(args, sample: Sample, **kwargs):
    custom_rm_path, rm_type = _resolve_reward_config(args, sample)

    if custom_rm_path is not None:
        rm_function = load_function(custom_rm_path)
        fn_args = compute_custom_function_config(args, custom_rm_path)
        return await rm_function(fn_args, sample, **kwargs)

    response = sample.response
    label = sample.label
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if rm_type.startswith("boxed_"):
        rm_type = rm_type[len("boxed_") :]
        if rm_type not in _BOXED_GRADES_ORIGINAL_RESPONSE:
            response = extract_boxed_answer(response) or ""
            if response and rm_type == "dapo":
                response = f"Answer: {response}"

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
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

    if args.custom_rm_path is not None:
        run_path = _run_custom_rm_path(args)
        row_paths = [_read_row_custom_rm_path(sample) for sample in samples]
        if rows_with_other_path := [i for i, path in enumerate(row_paths) if path not in (None, run_path)]:
            raise ValueError(
                f"metadata['custom_rm_path'] on group rows {rows_with_other_path} differs from --custom-rm-path, "
                "but --group-rm scores the whole group with that one batched function; drop one of the two"
            )
        rm_function = load_function(args.custom_rm_path)
        fn_args = compute_custom_function_config(args, args.custom_rm_path)
        return await rm_function(fn_args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
