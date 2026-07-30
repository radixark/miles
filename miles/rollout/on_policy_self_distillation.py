import logging
import math
from argparse import Namespace
from typing import Any

import aiohttp
import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

Prompt = str | list[dict[str, Any]]

_PRIVILEGED_CONTEXT = """

Here is a reference solution:
=== Reference Solution Begin ===
{label}
=== Reference Solution End ===

Understand the reference solution, then solve the original problem using independent reasoning rather than copying it.
""".strip(
    "\n"
)


def build_teacher_prompt(prompt: Prompt, label: str, metadata: dict[str, Any]) -> Prompt:
    del metadata
    if not isinstance(label, str) or not label.strip():
        raise ValueError("OPSD requires a non-empty text label containing the privileged reference solution.")

    privileged_context = _PRIVILEGED_CONTEXT.format(label=label)
    if isinstance(prompt, str):
        return f"{prompt}\n\n{privileged_context}"

    if not prompt or not all(isinstance(message, dict) for message in prompt):
        raise ValueError("The default OPSD teacher prompt builder requires a non-empty chat conversation.")
    if prompt[-1].get("role") != "user":
        raise ValueError("The default OPSD teacher prompt builder requires a conversation ending in a user message.")
    if any(message.get("role") == "tool" for message in prompt):
        raise ValueError("The default OPSD teacher prompt builder does not support tool messages.")
    if any(not isinstance(message.get("content"), str) for message in prompt):
        raise ValueError("The default OPSD teacher prompt builder supports text-only conversation content.")

    final_message = {
        **prompt[-1],
        "content": f"{prompt[-1]['content']}\n\n{privileged_context}",
    }
    return [*prompt[:-1], final_message]


def _score_payload(input_ids: list[int], *, top_k: int, temperature: float) -> dict[str, Any]:
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": top_k,
    }


def _extract_teacher_top_k(
    response: dict[str, Any],
    *,
    response_length: int,
    top_k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    try:
        values = response["meta_info"]["input_top_logprobs"]
    except (KeyError, TypeError) as exc:
        raise ValueError("OPSD teacher response is missing meta_info.input_top_logprobs.") from exc

    response_values = values[-response_length:] if response_length > 0 else []
    if len(response_values) != response_length:
        raise ValueError(
            f"OPSD teacher returned {len(response_values)} response positions, expected {response_length}."
        )

    token_ids = []
    scores = []
    for position, entries in enumerate(response_values):
        if entries is None or len(entries) != top_k:
            actual = 0 if entries is None else len(entries)
            raise ValueError(f"OPSD teacher position {position} returned top_k={actual}, expected {top_k}.")
        position_ids = [int(entry[1]) for entry in entries]
        position_scores = [float(entry[0]) for entry in entries]
        if len(set(position_ids)) != top_k:
            raise ValueError(f"OPSD teacher position {position} returned duplicate token ids.")
        if not all(math.isfinite(score) for score in position_scores):
            raise ValueError(f"OPSD teacher position {position} returned non-finite scores.")
        token_ids.append(position_ids)
        scores.append(position_scores)

    return token_ids, scores


async def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_secs: int | float | None,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    if sample.privileged_prompt_tokens is None:
        raise ValueError("OPSD SGLang scoring requires Sample.privileged_prompt_tokens.")

    response_tokens = sample.tokens[-sample.response_length :] if sample.response_length > 0 else []
    teacher_input_ids = [*sample.privileged_prompt_tokens, *response_tokens]
    payload = _score_payload(
        teacher_input_ids,
        top_k=args.opsd_teacher_top_k,
        temperature=args.rollout_temperature,
    )
    try:
        response = await _post_json(
            args.opsd_teacher_url,
            payload,
            timeout_secs=getattr(args, "sglang_router_request_timeout_secs", None),
        )
        token_ids, scores = _extract_teacher_top_k(
            response,
            response_length=sample.response_length,
            top_k=args.opsd_teacher_top_k,
        )
    except Exception:
        logger.exception(
            "OPSD teacher scoring failed: sample_index=%s teacher_input_length=%d response_length=%d endpoint=%s",
            sample.index,
            len(teacher_input_ids),
            sample.response_length,
            args.opsd_teacher_url,
        )
        raise

    return {"token_ids": token_ids, "scores": scores}


def post_process_rewards(
    args: Namespace,
    samples: list[Sample],
    **kwargs: Any,
) -> tuple[list[float], list[float]]:
    del kwargs
    for sample in samples:
        reward = sample.reward
        if not isinstance(reward, dict) or set(reward) != {"token_ids", "scores"}:
            raise ValueError("OPSD SGLang scoring must return compact token_ids and scores.")
        shape = (sample.response_length, args.opsd_teacher_top_k)
        sample.opsd_teacher_token_ids = torch.as_tensor(reward["token_ids"], dtype=torch.int64).reshape(shape)
        sample.opsd_teacher_scores = torch.as_tensor(reward["scores"], dtype=torch.float32).reshape(shape)
        sample.validate()

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
