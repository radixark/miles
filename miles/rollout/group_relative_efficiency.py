"""Quality-conditioned, group-relative trajectory efficiency shaping."""

from __future__ import annotations

import os
from typing import Any


def efficiency_weight(args: Any) -> float:
    value = getattr(args, "polar_group_relative_token_weight", None)
    if value is None:
        # Polar now owns task-level group shaping. Keep the adapter fallback
        # opt-in for compatibility with older Polar rollout servers.
        value = os.environ.get("POLAR_GROUP_RELATIVE_TOKEN_WEIGHT", "0.0")
    return max(0.0, float(value))


def trajectory_token_cost(samples: list[Any], indices: list[int]) -> float:
    """Read authoritative flow usage, falling back to train-sequence lengths."""
    input_weight = float(os.environ.get("POLAR_GROUP_RELATIVE_INPUT_WEIGHT", "0.05"))
    for index in indices:
        metadata = getattr(samples[index], "metadata", {})
        if not isinstance(metadata, dict):
            continue
        evaluation = (
            ((metadata.get("polar") or {}).get("trajectory_metadata") or {})
            .get("evaluation") or {}
        )
        usage = ((evaluation.get("raw_flow") or {}).get("token_usage") or {})
        if not isinstance(usage, dict) or not usage:
            continue
        output = _numeric(usage, "output", "output_tokens", "completion_tokens")
        input_tokens = _numeric(usage, "input", "input_tokens", "prompt_tokens")
        if output > 0.0 or input_tokens > 0.0:
            return output + input_weight * input_tokens

    return float(sum(
        max(0, int(getattr(samples[index], "response_length", 0) or 0))
        for index in indices
    ))


def shape_trajectory_means(
    means: dict[Any, float],
    costs: dict[Any, float],
    *,
    weight: float,
) -> dict[Any, float]:
    """Reward Pareto-efficient candidates without using a fixed token budget.

    A candidate wins a pair only when it is no worse in base quality and uses
    fewer tokens. The bounded bonus is proportional to remaining headroom, so
    it cannot turn a low-quality completion into a high-quality one.
    """
    if weight <= 0.0 or len(means) < 2:
        return dict(means)
    shaped: dict[Any, float] = {}
    denominator = len(means) - 1
    for key, quality in means.items():
        wins = sum(
            1 for other, other_quality in means.items()
            if other != key
            and quality >= other_quality
            and costs.get(key, 0.0) < costs.get(other, 0.0)
        )
        efficiency = wins / denominator
        bounded_quality = max(0.0, min(1.0, quality))
        shaped[key] = quality + weight * (1.0 - bounded_quality) * efficiency
    return shaped


def _numeric(values: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
    return 0.0


__all__ = ["efficiency_weight", "shape_trajectory_means", "trajectory_token_cost"]
