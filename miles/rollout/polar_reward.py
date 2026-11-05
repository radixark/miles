"""Polar reward adapter for Miles and Slime.

Thin Miles-side counterpart to ``slime_bridge/reward.py`` and
``slime_bridge/reward_post_process.py``.  A Polar trajectory already carries a
per-trace scalar ``score`` (embedded into each converted sample); the reward
module only needs to read it back.  No Polar core, slime_bridge, or downstream
harness code is touched or imported by this module.

Canonical entrypoints
---------------------
``custom_rm(args, sample_or_samples) -> float | list[float]``
    The canonical Miles ``--custom-rm-path`` entrypoint. Miles' runtime loads
    the configured symbol for both single-sample and batched reward paths, so
    this coroutine accepts either shape and preserves the corresponding return
    shape. Register this symbol as your ``--custom-rm-path``.

``compute_reward(args, sample) -> float``
    The synchronous pure equivalent of ``custom_rm``.  Holds the actual reward
    extraction logic and returns a plain ``float``, so any synchronous consumer
    (or the documented ``def custom_rm(args, sample) -> float`` contract) can call
    it directly without ``asyncio``.

``reward_func(args, sample_or_samples, **kwargs)``
    Slime-compatible async reward hook mirroring ``slime_bridge.reward.reward_func``,
    kept for drop-in parity with the Slime adapter contract.  Accepts either a
    single sample or a list and returns a ``{polar_reward_key: float}`` dict (or a
    list of such dicts).  Delegates to ``compute_reward`` to avoid duplication.

``post_process_rewards(args, samples) -> (raw_rewards, rewards)``
    Port of ``slime_bridge.reward_post_process.post_process_rewards``: leave-one-
    trajectory-out advantage normalization keyed by ``Sample.group_id`` so that a
    trajectory with a variable number of traces counts as one gradient unit.

``custom_rm``, ``compute_reward``, and ``reward_func`` all defensively read the
Polar sample shape: a nested sample dict whose ``polar_reward_key`` (default
``"score"``) holds the scalar reward.  Purely numeric rewards and a dict-of-many
numeric values are also handled, matching Slime's extraction behaviour.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

from miles.rollout.group_relative_efficiency import (
    efficiency_weight,
    shape_trajectory_means,
    trajectory_token_cost,
)

logger = logging.getLogger(__name__)


def _reward_key(args: Any) -> str:
    return str(getattr(args, "polar_reward_key", getattr(args, "reward_key", "score")))


def _extract_reward(sample: Any, reward_key: str) -> float:
    """Return the Polar score embedded in a single sample.

    A Polar-converted sample stores its reward under the ``polar_reward_key``
    field (default ``"score"``), either as a direct attribute or inside the
    nested ``sample["sample"]["response"]["reward"]`` dict that Polar produces.
    Fallbacks mirror Slime's ``_extract_reward``: a plain numeric reward, a
    ``dict`` whose key matches, a ``"score"`` key, or any single numeric value.
    """
    reward = getattr(sample, "reward", None)

    if reward is None and isinstance(sample, dict):
        # Polar nests the trace under sample["sample"]; dig down defensively to
        # the innermost dict(s) that actually carry a reward value.
        candidate: Any = sample
        for _ in range(8):
            if not isinstance(candidate, dict):
                break
            if "reward" in candidate:
                reward = candidate["reward"]
                break
            inner = candidate.get("sample")
            if isinstance(inner, dict):
                candidate = inner
            else:
                break

    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
        for value in reward.values():
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0


def compute_reward(args: Any, sample: Any) -> float:
    """Synchronous pure reward extraction for a single Polar sample.

    This holds the actual extraction logic shared by the async entrypoints.
    Returns a plain ``float`` so any synchronous consumer can use it directly, per
    the documented ``def custom_rm(args, sample) -> float`` contract.
    """
    reward_key = _reward_key(args)
    reward = _extract_reward(sample, reward_key)
    logger.debug("polar reward -> %s (key=%s)", reward, reward_key)
    return float(reward)


async def custom_rm(
    args: Any, sample_or_samples: Any, **kwargs: Any
) -> float | list[float]:
    """Miles ``--custom-rm-path`` entrypoint for single and batched calls.

    ``rm_hub.async_rm`` passes one sample, while
    ``rm_hub.batched_async_rm`` passes the complete sample list to the same
    configured callable. Preserve that distinction in the return value so both
    Miles paths receive the shape they expect.
    """
    del kwargs
    if isinstance(sample_or_samples, list):
        return [compute_reward(args, sample) for sample in sample_or_samples]
    return compute_reward(args, sample_or_samples)


async def reward_func(args: Any, sample_or_samples: Any, **kwargs: Any) -> Any:
    """Slime-compatible async reward hook (drop-in parity with slime_bridge).

    Accepts a single sample or a list of samples.  Returns a ``{reward_key: float}``
    dict for a single sample, or a list of such dicts for a batch, mirroring the
    Slime adapter contract.  Delegates to ``compute_reward`` so both paths share the
    same extraction logic as ``custom_rm``.
    """
    del kwargs
    reward_key = _reward_key(args)
    if isinstance(sample_or_samples, list):
        return [{reward_key: compute_reward(args, sample)} for sample in sample_or_samples]
    return {reward_key: compute_reward(args, sample_or_samples)}


def post_process_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Miles ``--custom-reward-post-process-path`` hook. Returns ``(raw, rewards)``.

    Port of ``slime_bridge.reward_post_process.post_process_rewards``: builds a
    per-trace advantage against a leave-one-trajectory-out baseline derived from
    other trajectories sharing the same prompt group, so a Polar trajectory that
    fans out into multiple traces still counts as one gradient unit.  Honors the
    same ``rewards_normalization``, ``advantage_estimator``, and ``grpo_std_normalization``
    switches as Slime.
    """
    raw_rewards = [float(_extract_reward(sample, _reward_key(args))) for sample in samples]

    if not getattr(args, "rewards_normalization", True):
        return raw_rewards, list(raw_rewards)

    estimator = getattr(args, "advantage_estimator", None)
    if estimator not in ("grpo", "gspo", "reinforce_plus_plus_baseline"):
        return raw_rewards, list(raw_rewards)

    std_norm = estimator in ("grpo", "gspo") and bool(getattr(args, "grpo_std_normalization", False))

    traj_sample_indices: dict[Any, list[int]] = {}
    traj_valid_rewards: dict[Any, list[float]] = {}
    traj_failed: dict[Any, bool] = {}
    group_keys: dict[Any, list[Any]] = {}

    for i, sample in enumerate(samples):
        group_idx, key = _trajectory_key(sample, i)
        if key not in traj_sample_indices:
            traj_sample_indices[key] = []
            traj_valid_rewards[key] = []
            traj_failed[key] = False
            group_keys.setdefault(group_idx, []).append(key)
        traj_sample_indices[key].append(i)
        if _is_failed_trajectory(sample):
            traj_failed[key] = True
        elif _has_trainable_tokens(sample):
            traj_valid_rewards[key].append(raw_rewards[i])

    normalized_by_sample = [1e-8] * len(samples)
    efficiency_bonus_by_key: dict[Any, float] = {}
    efficiency_weight_value = efficiency_weight(args)
    for keys in group_keys.values():
        valid_keys = [key for key in keys if not traj_failed[key] and traj_valid_rewards[key]]
        traj_mean = {
            key: sum(traj_valid_rewards[key]) / len(traj_valid_rewards[key]) for key in valid_keys
        }
        traj_cost = {
            key: trajectory_token_cost(samples, traj_sample_indices[key])
            for key in valid_keys
        }
        shaped_mean = shape_trajectory_means(
            traj_mean,
            traj_cost,
            weight=efficiency_weight_value,
        )

        for key in keys:
            if key not in shaped_mean:
                continue
            efficiency_bonus_by_key[key] = shaped_mean[key] - traj_mean[key]
            other_means = [
                shaped_mean[other_key]
                for other_key in valid_keys
                if other_key != key
            ]
            baseline = sum(other_means) / len(other_means) if other_means else 0.0
            scale = _loo_scale(other_means) if std_norm else 1.0
            for sample_index in traj_sample_indices[key]:
                normalized_by_sample[sample_index] = (
                    shaped_mean[key] - baseline
                ) / scale

    for key, indices in traj_sample_indices.items():
        if key not in traj_mean:
            continue
        metadata = getattr(samples[indices[0]], "metadata", None)
        if not isinstance(metadata, dict):
            continue
        polar_metadata = metadata.setdefault("polar", {})
        polar_metadata["group_relative_efficiency"] = {
            "enabled": efficiency_weight_value > 0.0,
            "weight": efficiency_weight_value,
            "base_reward": traj_mean[key],
            "bonus": efficiency_bonus_by_key.get(key, 0.0),
            "shaped_reward": traj_mean[key] + efficiency_bonus_by_key.get(key, 0.0),
            "token_cost": traj_cost.get(key, 0.0),
        }

    normalized_by_sample = [r if abs(r) > 1e-10 else 1e-8 for r in normalized_by_sample]
    return raw_rewards, normalized_by_sample


def _trajectory_key(sample: Any, sample_position: int) -> tuple[Any, Any]:
    group_idx = _key_value(getattr(sample, "group_index", None), -1)
    traj_idx = getattr(sample, "rollout_id", None)
    if traj_idx is None:
        traj_idx = getattr(sample, "index", None)
    return group_idx, (group_idx, _key_value(traj_idx, sample_position))


def _key_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _loo_scale(other_means: list[float]) -> float:
    if len(other_means) <= 1:
        return 1.0
    return statistics.stdev(other_means) + 1e-6


def _has_trainable_tokens(sample: Any) -> bool:
    if bool(getattr(sample, "remove_sample", False)):
        return False
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0) > 0
    return any(int(value) != 0 for value in loss_mask)


def _is_failed_trajectory(sample: Any) -> bool:
    """True if the sample's status marks it as agent ERROR or TIMEOUT."""
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    return name.upper() in ("FAILED", "ABORTED")


__all__ = [
    "custom_rm",
    "compute_reward",
    "reward_func",
    "post_process_rewards",
    "_extract_reward",
    "_trajectory_key",
    "_key_value",
    "_loo_scale",
    "_has_trainable_tokens",
    "_is_failed_trajectory",
]
