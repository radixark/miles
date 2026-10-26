"""Per-sample filters for --rollout-sample-filter-path.

A sample filter is called as ``fn(args, data)`` with ``data: list[list[Sample]]`` and marks
samples to leave out of the loss by setting ``sample.remove_sample = True``. Removed samples
still count toward their group's reward normalization.
"""

import logging
from argparse import Namespace

from miles.rollout.filter_hub.base_types import iter_samples
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# ``metadata["exit_status"]`` values agent environments use for a trajectory that was cut off
# rather than finished (Harbor's names for the wall-clock and context limits).
CUT_OFF_EXIT_STATUSES = frozenset({"TimeLimitExceeded", "SequenceLengthLimitExceeded"})

_warned_dict_reward = False


def is_cut_off(sample: Sample) -> bool:
    return (
        sample.status == Sample.Status.TRUNCATED or (sample.metadata or {}).get("exit_status") in CUT_OFF_EXIT_STATUSES
    )


def mask_truncated_zero_reward(args: Namespace, data: list[list[Sample]]) -> None:
    """Leave cut-off trajectories that earned no reward out of the loss.

    A trajectory stopped by a length or time limit did not get to finish, so a reward of 0
    says little about the actions it took; training on it penalizes long exploration itself
    (DeepSWE's "compact filtering"). Cut-off trajectories with a positive reward stay: the
    grader may still pass a patch the agent wrote before the limit hit.
    """
    global _warned_dict_reward
    total = masked = 0
    for sample in iter_samples(data):
        total += 1
        if not args.reward_key and isinstance(sample.reward, dict):
            # No scalar to compare without --reward-key; leave the sample in the loss.
            if not _warned_dict_reward:
                logger.warning(
                    "mask_truncated_zero_reward: rewards are dicts and --reward-key is not set, "
                    "so those samples are never masked"
                )
                _warned_dict_reward = True
            continue
        reward = sample.get_reward_value(args) if sample.reward is not None else None
        if is_cut_off(sample) and (reward is None or reward <= 0):
            sample.remove_sample = True
            masked += 1
    if masked:
        logger.info(f"mask_truncated_zero_reward: left {masked}/{total} cut-off zero-reward samples out of the loss")
