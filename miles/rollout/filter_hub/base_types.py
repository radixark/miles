from collections import defaultdict
from dataclasses import dataclass

from miles.utils.types import Sample


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None


def call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


def _flatten_samples(group) -> list[Sample]:
    return [s for item in group for s in (item if isinstance(item, list) else [item])]


def group_has_aborted(group) -> bool:
    return any(s.status == Sample.Status.ABORTED for s in _flatten_samples(group))


def aborted_exit_status(group) -> str:
    """The cause the first aborted sample recorded (agentic_tool_call sets it), else "unknown"."""
    for s in _flatten_samples(group):
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

    def on_aborted_group_drop(self, group):
        self._aborted_drop_reason_count[aborted_exit_status(group)] += 1

    def collect(self):
        return {
            **{
                f"rollout/dynamic_filter/drop_{reason}": count
                for reason, count in self._dynamic_filter_drop_reason_count.items()
            },
            **{f"rollout/aborted/drop_{reason}": count for reason, count in self._aborted_drop_reason_count.items()},
        }
