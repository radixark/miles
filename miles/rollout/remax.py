"""Greedy reward baselines for prompt-group rollout generation."""

import uuid
from argparse import Namespace
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from miles.utils.types import Sample


async def generate_remax_baseline(
    args: Namespace,
    group: list[Sample],
    sampling_params: dict[str, Any],
    generate_fn: Callable[..., Awaitable[Sample]],
) -> bool:
    """Attach one greedy reward to the training samples; reject an unscored group."""
    baseline = deepcopy(group[0])
    baseline.reset_for_retry()
    baseline.trainer_model_id = group[0].trainer_model_id
    baseline.kv_cache_namespace = group[0].kv_cache_namespace
    baseline.index = -(group[0].index or 0) - 1
    baseline.rollout_id = baseline.index
    baseline.routing_key = str(uuid.uuid4())
    baseline.metadata = {**(baseline.metadata or {}), "remax_baseline": True}
    baseline.remove_sample = True
    # This response supplies only a reward, so skip training-only sampling replay and lineage.
    baseline = await generate_fn(baseline, {**sampling_params, "temperature": 0.0}, evaluation=True)
    if not isinstance(baseline, Sample):
        raise ValueError("remax greedy generation must return one Sample")
    reward = baseline.get_reward_value(args) if baseline.reward is not None else None
    if baseline.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or reward is None:
        for sample in group:
            sample.status = Sample.Status.ABORTED
        return False
    for sample in group:
        sample.metadata = {**(sample.metadata or {}), "remax_baseline_reward": reward}
    return True
