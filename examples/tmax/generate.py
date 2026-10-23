"""TMax reward functions, and the reward post-process of pure on-policy distillation.

The reward is computed in the sandbox by the task's tests (tmax_agent_function.run)
and rides back in ``sample.metadata["reward"]``. ``None`` there means the sandbox
failed for a non-policy reason; returning it lets the missing-reward filter drop
the group instead of training on a false 0.
"""

import asyncio
import logging
import os

from miles.rollout import on_policy_distillation
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# A teacher request that errors or times out is retried; after this many attempts it fails the rollout, since a teacher
# that stopped answering would otherwise leave every step without its OPD signal.
_TEACHER_ATTEMPTS = 3


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:
    if isinstance(samples, list):
        return [s.metadata.get("reward") for s in samples]
    return samples.metadata.get("reward")


async def reward_func_with_teacher(
    args, samples: Sample | list[Sample], **kwargs
) -> float | None | list[float | None]:
    """``--custom-rm-path`` for OPD with an SGLang teacher (``--opd-type sglang``): ``reward_func``'s task reward, and
    the teacher's log-prob of every response token in ``sample.teacher_log_probs``, which Miles trains on as the
    Megatron teacher's.

    Miles' own SGLang-mode reward function returns the teacher's response as the reward and its post-process sets
    every reward to 0; this keeps the task reward, so the missing-reward filter, the metrics, additive OPD and
    ``opd_pure_rewards`` work as with the Megatron teacher. The teacher request is Miles' (``--rm-url``, or
    ``--opd-teacher-urls`` routed by sample metadata). A sample without a task reward is not scored: its group is
    dropped.
    """
    batch = samples if isinstance(samples, list) else [samples]
    await asyncio.gather(*(_score_with_teacher(args, s) for s in batch if s.metadata.get("reward") is not None))
    return await reward_func(args, samples)


async def _score_with_teacher(args, sample: Sample) -> None:
    timeout_s = float(os.environ.get("TMAX_OPD_TEACHER_TIMEOUT_S", "1800"))
    for attempt in range(1, _TEACHER_ATTEMPTS + 1):
        try:
            response = await asyncio.wait_for(on_policy_distillation.reward_func(args, sample), timeout=timeout_s)
            break
        except Exception as e:  # an HTTP error, a dropped connection, the timeout
            if attempt == _TEACHER_ATTEMPTS:
                raise RuntimeError(f"the OPD teacher failed {attempt} times on sample {sample.index}") from e
            logger.warning(f"OPD teacher request for sample {sample.index} failed (attempt {attempt}): {e!r}")
            await asyncio.sleep(10 * attempt)
    sample.teacher_log_probs = _response_log_probs(response, sample)


def _response_log_probs(response: dict, sample: Sample) -> list[float]:
    """The teacher's log-prob of each response token. SGLang scores every input token after the first
    (``logprob_start_len=0``), as Miles' ``post_process_rewards`` reads it; the scored ids must be the sample's."""
    n = sample.response_length
    scored = response["meta_info"]["input_token_logprobs"][1:][-n:] if n else []
    if [entry[1] for entry in scored] != (sample.tokens[-n:] if n else []):
        raise ValueError(f"the OPD teacher scored other tokens than sample {sample.index}'s response")
    return [float(entry[0]) for entry in scored]


def opd_pure_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """``--custom-reward-post-process-path`` for pure OPD, open-instruct's --opd_pure.

    Returns (raw rewards, trained rewards). Every trained reward is 0, so GRPO's advantage is 0 and the advantage
    is Miles' OPD term alone, -opd_kl_coef * (log p_rollout - log p_teacher) per token. The raw rewards are the task
    rewards, which Miles logs as rollout/raw_reward. The missing-reward filter has already dropped groups without one.
    """
    return [s.get_reward_value(args) for s in samples], [0.0] * len(samples)
