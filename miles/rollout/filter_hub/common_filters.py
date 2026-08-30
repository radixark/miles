from argparse import Namespace

from miles.rollout.filter_hub.base_types import FilterOutput, iter_samples
from miles.utils.types import Sample

Group = list[Sample | list[Sample]]


def check_no_aborted(args: Namespace, samples: Group, **kwargs) -> FilterOutput:
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(sample.status == Sample.Status.ABORTED for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="group_has_aborted")
    return FilterOutput(keep=True)


def group_staleness(group: Group, current_version: int | None) -> int | None:
    versions = [version for sample in iter_samples(group) if (version := sample.oldest_weight_version) is not None]
    oldest = min(versions) if versions else None
    if oldest is None or current_version is None:
        return None
    return current_version - oldest
