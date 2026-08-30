from argparse import Namespace

import torch

from miles.rollout.filter_hub.base_types import FilterOutput, call_dynamic_filter, iter_samples
from miles.utils.types import Sample

Group = list[Sample | list[Sample]]


def apply_preput_filters(args: Namespace, dynamic_filter, samples: Group, **kwargs) -> FilterOutput:
    output = check_no_aborted(args, samples, **kwargs)
    if not output.keep:
        return output

    output = check_no_missing_reward(args, samples, **kwargs)
    if not output.keep:
        return output

    return call_dynamic_filter(dynamic_filter, args, samples, **kwargs)


def check_no_aborted(args: Namespace, samples: Group, **kwargs) -> FilterOutput:
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(sample.status == Sample.Status.ABORTED for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="group_has_aborted")
    return FilterOutput(keep=True)


def check_no_missing_reward(args: Namespace, samples: Group, **kwargs) -> FilterOutput:
    if any(sample.reward is None or sample.get_reward_value(args) is None for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="group_has_missing_reward")
    return FilterOutput(keep=True)


def check_reward_nonzero_std(args, samples: list[Sample | list[Sample]], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in iter_samples(samples)]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return FilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def group_staleness(group: Group, current_version: int | None) -> int | None:
    versions = [version for sample in iter_samples(group) if (version := sample.oldest_weight_version) is not None]
    oldest = min(versions) if versions else None
    if oldest is None or current_version is None:
        return None
    return current_version - oldest
