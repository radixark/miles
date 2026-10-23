"""TMax reward function, and the reward post-process of pure on-policy distillation.

The reward is computed in the sandbox by the task's tests (tmax_agent_function.run)
and rides back in ``sample.metadata["reward"]``. ``None`` there means the sandbox
failed for a non-policy reason; returning it lets the missing-reward filter drop
the group instead of training on a false 0.
"""

from miles.utils.types import Sample


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:
    if isinstance(samples, list):
        return [s.metadata.get("reward") for s in samples]
    return samples.metadata.get("reward")


def opd_pure_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """``--custom-reward-post-process-path`` for pure OPD, open-instruct's --opd_pure.

    Returns (raw rewards, trained rewards). Every trained reward is 0, so GRPO's advantage is 0 and the advantage
    is Miles' OPD term alone, -opd_kl_coef * (log p_rollout - log p_teacher) per token. The raw rewards are the task
    rewards, which Miles logs as rollout/raw_reward. The missing-reward filter has already dropped groups without one.
    """
    return [s.get_reward_value(args) for s in samples], [0.0] * len(samples)
