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


def group_has_aborted(group: list[Sample | list[Sample]]) -> bool:
    return any(s.status == Sample.Status.ABORTED for s in iter_samples(group))


def aborted_exit_status(group: list[Sample | list[Sample]]) -> str:
    """The cause the first aborted sample recorded (agentic_tool_call sets it), else "unknown"."""
    for s in iter_samples(group):
        if s.status == Sample.Status.ABORTED:
            return str(s.metadata.get("exit_status") or "unknown")
    return "unknown"


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._aborted_drop_reason_count = defaultdict(lambda: 0)

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def on_aborted_group_drop(self, group: list[Sample | list[Sample]]):
        self._aborted_drop_reason_count[aborted_exit_status(group)] += 1

    def collect(self):
        return {
            **{
                f"rollout/dynamic_filter/drop_{reason}": count
                for reason, count in self._dynamic_filter_drop_reason_count.items()
            },
            **{f"rollout/aborted/drop_{reason}": count for reason, count in self._aborted_drop_reason_count.items()},
        }
