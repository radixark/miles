import logging

import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

AGENTIC_INTERMEDIATE_TRUNCATED_KEY = "agentic_intermediate_truncated"

__all__ = [
    "check_reward_nonzero_std",
    "check_no_aborted",
    "check_no_aborted_and_reward_nonzero_std",
    "check_no_aborted_or_truncated",
    "check_no_aborted_or_truncated_and_reward_nonzero_std",
]

logger = logging.getLogger(__name__)


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    flat_samples = list(_flatten_samples(samples))
    rewards = [sample.get_reward_value(args) for sample in flat_samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    if not keep:
        logger.info("[dynamic-filter] drop reason=zero_std rewards=%s", rewards)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def _flatten_samples(samples):
    """Flatten samples that may contain nested lists (from --generate-multi-samples)."""
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def check_no_aborted(args, samples: list[Sample], **kwargs):
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    flat_samples = list(_flatten_samples(samples))
    if any(s.status == Sample.Status.ABORTED for s in flat_samples):
        logger.info(
            "[dynamic-filter] drop reason=group_has_aborted statuses=%s rewards=%s",
            [str(s.status) for s in flat_samples],
            [s.get_reward_value(args) for s in flat_samples],
        )
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def check_no_aborted_or_truncated(args, samples: list[Sample], **kwargs):
    """Reject groups with aborted or truncated samples.

    Agentic multi-turn truncation is a terminal trajectory state. It can contain
    partial, logprob-valid tokens, but it should not be trained as a normal SWE
    rollout sample.
    """
    flat_samples = list(_flatten_samples(samples))
    if any(s.status == Sample.Status.ABORTED for s in flat_samples):
        logger.info(
            "[dynamic-filter] drop reason=group_has_aborted statuses=%s rewards=%s",
            [str(s.status) for s in flat_samples],
            [s.get_reward_value(args) for s in flat_samples],
        )
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")

    if any(s.metadata.get(AGENTIC_INTERMEDIATE_TRUNCATED_KEY) for s in flat_samples):
        logger.info(
            "[dynamic-filter] drop reason=group_has_agentic_intermediate_truncated statuses=%s rewards=%s",
            [str(s.status) for s in flat_samples],
            [s.get_reward_value(args) for s in flat_samples],
        )
        return DynamicFilterOutput(keep=False, reason="group_has_agentic_intermediate_truncated")

    if any(s.status == Sample.Status.TRUNCATED for s in flat_samples):
        logger.info(
            "[dynamic-filter] drop reason=group_has_truncated statuses=%s rewards=%s",
            [str(s.status) for s in flat_samples],
            [s.get_reward_value(args) for s in flat_samples],
        )
        return DynamicFilterOutput(keep=False, reason="group_has_truncated")

    return DynamicFilterOutput(keep=True)


def check_no_aborted_and_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """Reject aborted groups and groups that cannot produce a GRPO advantage."""
    flat_samples = list(_flatten_samples(samples))
    no_aborted = check_no_aborted(args, samples, **kwargs)
    if not no_aborted.keep:
        return no_aborted
    reward_std = check_reward_nonzero_std(args, samples, **kwargs)
    if reward_std.keep:
        logger.info(
            "[dynamic-filter] keep reason=no_aborted_and_reward_nonzero_std rewards=%s statuses=%s",
            [s.get_reward_value(args) for s in flat_samples],
            [str(s.status) for s in flat_samples],
        )
    return reward_std


def check_no_aborted_or_truncated_and_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """Reject terminal/partial trajectories and zero-std reward groups."""
    flat_samples = list(_flatten_samples(samples))
    healthy = check_no_aborted_or_truncated(args, samples, **kwargs)
    if not healthy.keep:
        return healthy
    reward_std = check_reward_nonzero_std(args, samples, **kwargs)
    if reward_std.keep:
        logger.info(
            "[dynamic-filter] keep reason=no_aborted_or_truncated_and_reward_nonzero_std rewards=%s statuses=%s",
            [s.get_reward_value(args) for s in flat_samples],
            [str(s.status) for s in flat_samples],
        )
    return reward_std
