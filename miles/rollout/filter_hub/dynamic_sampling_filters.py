import torch

from miles.rollout.filter_hub.base_types import FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import check_no_aborted
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_no_aborted"]


def check_reward_nonzero_std(args, samples: list[Sample | list[Sample]], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in iter_samples(samples)]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return FilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
