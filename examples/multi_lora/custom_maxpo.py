"""Centered max@K advantages via conditional and leave-one-out expected maxima."""

import math

import numpy as np


def compute_mean_maxpo_advantages(
    rewards: np.ndarray, k: int, mean_weight: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Mix mean and max@K policy gradients with coefficients for a batch-mean loss."""
    n = len(rewards)
    if not 1 <= k < n:
        raise ValueError("MaxPO requires 1 <= k < group size")
    if not np.isfinite(rewards).all() or not 0 <= mean_weight <= 1:
        raise ValueError("MaxPO requires finite rewards and a mean weight in [0, 1]")
    order = np.argsort(rewards, kind="stable")
    ranked = rewards[order]
    included = np.zeros(n)
    excluded = np.zeros(n)
    include_denominator = math.comb(n - 1, k - 1)
    exclude_denominator = math.comb(n - 1, k)
    for i in range(n):
        included[i] = ranked[i] * (math.comb(i, k - 1) / include_denominator)
        for j in range(n):
            if j == i:
                continue
            lower = j - int(i < j)
            excluded[i] += ranked[j] * (math.comb(lower, k - 1) / exclude_denominator)
            if j > i and k > 1:
                included[i] += ranked[j] * (math.comb(j - 1, k - 2) / include_denominator)
    max_advantages = k * (included - excluded)
    mean_advantages = (ranked - ranked.mean()) * n / (n - 1)
    advantages = np.empty(n)
    advantages[order] = mean_weight * mean_advantages + (1 - mean_weight) * max_advantages
    expected_max = float(included.mean())
    return advantages, {
        "maxpo/k": float(k),
        "maxpo/group_size": float(n),
        "maxpo/mean_weight": mean_weight,
        "maxpo/expected_max_reward": expected_max,
        "maxpo/joint_reward": mean_weight * float(rewards.mean()) + (1 - mean_weight) * expected_max,
        "maxpo/mean_advantage_std": float(mean_advantages.std()),
        "maxpo/max_advantage_std": float(max_advantages.std()),
        "maxpo/advantage_mean": float(advantages.mean()),
        "maxpo/advantage_std": float(advantages.std()),
    }
