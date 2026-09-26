from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from miles.utils.types import Sample


@dataclass
class FilterOutput:
    keep: bool
    reason: str | None = None


DynamicFilterOutput = FilterOutput


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    if fn is None:
        return FilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, FilterOutput):
        output = FilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._aborted_trajectories_filtered = 0
        self._partial_groups_retained = 0

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def on_aborted_trajectories(self, count: int, *, group_retained: bool) -> None:
        self._aborted_trajectories_filtered += count
        self._partial_groups_retained += int(group_retained)

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._aborted_trajectories_filtered:
            metrics["rollout/aborted_trajectories_filtered"] = self._aborted_trajectories_filtered
            metrics["rollout/partial_groups_retained"] = self._partial_groups_retained
        return metrics
