"""Prompt-group reward shaping before advantage normalization."""

import logging
import math
from argparse import Namespace

from miles.utils.types import Sample


def is_correct_reward(reward: float) -> bool:
    """Only an exact raw reward of one denotes a correct trajectory."""
    return reward == 1.0

logger = logging.getLogger(__name__)


def validate_length_penalty_args(args: Namespace) -> None:
    coeff = args.length_penalty_coeff
    minimum = args.length_penalty_min_tokens
    if not math.isfinite(coeff) or coeff < 0:
        raise ValueError("--length-penalty-coeff must be finite and non-negative")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("--length-penalty-min-tokens must be a positive integer")
    if coeff > 0:
        if args.custom_reward_post_process_path:
            raise ValueError("--length-penalty-coeff cannot be combined with --custom-reward-post-process-path")
        if args.custom_convert_samples_to_train_data_path:
            raise ValueError(
                "--length-penalty-coeff cannot be combined with --custom-convert-samples-to-train-data-path"
            )


def reward_group_segments(
    args: Namespace, samples: list[Sample], prompt_group_sizes: list[int] | None = None
) -> list[list[int]]:
    """Return flattened row indices for each prompt, including interleaved batches."""
    if prompt_group_sizes is not None:
        assert sum(prompt_group_sizes) == len(
            samples
        ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(samples)} rewards"
        groups: list[list[int]] = []
        start = 0
        for size in prompt_group_sizes:
            if size > 0:
                groups.append(list(range(start, start + size)))
            start += size
        return groups
    if all(sample.group_index is not None for sample in samples):
        by_prompt: dict[int, list[int]] = {}
        for row, sample in enumerate(samples):
            by_prompt.setdefault(int(sample.group_index), []).append(row)
        return list(by_prompt.values())
    if len(samples) == args.n_samples_per_prompt * args.rollout_batch_size:
        return [
            list(range(start, start + args.n_samples_per_prompt))
            for start in range(0, len(samples), args.n_samples_per_prompt)
        ]
    return [list(range(len(samples)))]


def rollout_segments(samples: list[Sample], rows: list[int]) -> list[list[int]]:
    """Collect sibling training rows belonging to one generated trajectory."""
    by_rollout: dict[int | tuple[str, int], list[int]] = {}
    for row in rows:
        sample = samples[row]
        key = sample.rollout_id if sample.rollout_id is not None else sample.index
        by_rollout.setdefault(key if key is not None else ("row", row), []).append(row)
    return list(by_rollout.values())


def apply_length_penalty(
    args: Namespace,
    samples: list[Sample],
    rewards: list[float],
    prompt_groups: list[list[int]],
    *,
    log_penalties: bool = False,
) -> list[float]:
    """Shape correct rollout rewards without mutating samples or raw rewards."""
    coeff = args.length_penalty_coeff
    if coeff == 0:
        return rewards
    minimum = args.length_penalty_min_tokens
    shaped = list(rewards) if coeff else rewards
    for rows in prompt_groups:
        rollouts: list[tuple[list[int], int]] = []
        correct: list[tuple[list[int], int]] = []
        for siblings in rollout_segments(samples, rows):
            reward = rewards[siblings[0]]
            if coeff and any(rewards[row] != reward for row in siblings[1:]):
                raise ValueError("all samples in a rollout must share one reward")
            length = sum(
                0 if samples[row].remove_sample else samples[row].effective_response_length for row in siblings
            )
            rollouts.append((siblings, length))
            if is_correct_reward(reward):
                if length < 0:
                    raise ValueError("generated token count must be non-negative")
                if length > 0:
                    correct.append((siblings, length))
        reference = max(minimum, min(length for _, length in correct)) if correct else None
        multipliers = {siblings[0]: 1.0 for siblings, _ in rollouts}
        for siblings, length in correct:
            reward = max(1.0, length / reference) ** (-coeff)
            multipliers[siblings[0]] = reward
            for row in siblings:
                shaped[row] = reward
        if log_penalties:
            for siblings, length in rollouts:
                row = siblings[0]
                sample = samples[row]
                logger.info(
                    "Length penalty: session_id=%s rollout_id=%s generated_tokens=%d "
                    "multiplier=%.12g reward_reduction=%.12g raw_reward=%.12g shaped_reward=%.12g",
                    sample.metadata.get("session_id"),
                    sample.rollout_id if sample.rollout_id is not None else sample.index,
                    length,
                    multipliers[row],
                    rewards[row] - shaped[row],
                    rewards[row],
                    shaped[row],
                )
    return shaped
