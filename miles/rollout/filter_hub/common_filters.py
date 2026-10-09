from argparse import Namespace
from dataclasses import dataclass

import torch

from miles.rollout.filter_hub.base_types import FilterOutput, call_dynamic_filter, iter_samples
from miles.utils.types import Sample

Group = list[Sample | list[Sample]]


@dataclass(frozen=True)
class GroupWeightVersionStats:
    sample_count: int
    versioned_sample_count: int
    versioned_token_count: int
    token_version_sum: int
    oldest_version: int | None
    newest_version: int | None

    def oldest_lag(self, current_version: int | None) -> int | None:
        if self.oldest_version is None or current_version is None:
            return None
        return current_version - self.oldest_version

    def newest_lag(self, current_version: int | None) -> int | None:
        if self.newest_version is None or current_version is None:
            return None
        return current_version - self.newest_version

    def token_weighted_lag(self, current_version: int | None) -> float | None:
        if current_version is None or self.versioned_token_count == 0:
            return None
        return current_version - self.token_version_sum / self.versioned_token_count


def group_weight_version_stats(group: Group) -> GroupWeightVersionStats:
    sample_count = 0
    versioned_sample_count = 0
    versioned_token_count = 0
    token_version_sum = 0
    oldest_version = None
    newest_version = None

    for sample in iter_samples(group):
        sample_count += 1
        sample_has_version = False
        for span in sample.all_weight_version_spans:
            if not str(span.version).isdigit():
                continue
            version = int(span.version)
            token_count = span.abs_end - span.abs_start
            versioned_token_count += token_count
            token_version_sum += version * token_count
            oldest_version = version if oldest_version is None else min(oldest_version, version)
            newest_version = version if newest_version is None else max(newest_version, version)
            sample_has_version = True
        versioned_sample_count += int(sample_has_version)

    return GroupWeightVersionStats(
        sample_count=sample_count,
        versioned_sample_count=versioned_sample_count,
        versioned_token_count=versioned_token_count,
        token_version_sum=token_version_sum,
        oldest_version=oldest_version,
        newest_version=newest_version,
    )


def apply_preput_filters(args: Namespace, dynamic_filter, samples: Group, **kwargs) -> FilterOutput:
    output = apply_aborted_filter(args, samples, **kwargs)
    if not output.keep:
        return output

    output = apply_missing_reward_filter(args, samples, **kwargs)
    if not output.keep:
        return output

    return call_dynamic_filter(dynamic_filter, args, samples, **kwargs)


def apply_aborted_filter(args: Namespace, samples: Group, **kwargs) -> FilterOutput:
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(sample.status == Sample.Status.ABORTED for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="group_has_aborted")
    return FilterOutput(keep=True)


def apply_missing_reward_filter(args: Namespace, samples: Group, **kwargs) -> FilterOutput:
    if any(sample.reward is None or sample.get_reward_value(args) is None for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="group_has_missing_reward")
    return FilterOutput(keep=True)


def apply_reward_nonzero_std_filter(args, samples: list[Sample | list[Sample]], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in iter_samples(samples)]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return FilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def group_staleness(group: Group, current_version: int | None) -> int | None:
    return group_weight_version_stats(group).oldest_lag(current_version)
