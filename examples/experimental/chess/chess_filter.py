"""Dynamic-sampling filter for chess rollout groups."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample


def _flatten_samples(samples: Iterable[Sample | list[Sample]]) -> Iterator[Sample]:
    for item in samples:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _chess_outcome(sample: Sample) -> object:
    result = sample.metadata.get("chess_result")
    return result.get("outcome") if isinstance(result, Mapping) else None


def check_chess_group(
    args: Any,
    samples: list[Sample | list[Sample]],
    **kwargs: Any,
) -> DynamicFilterOutput:
    """Reject groups containing aborted, unscored, or infrastructure-error games."""

    del args, kwargs
    flattened = list(_flatten_samples(samples))
    if any(sample.status == Sample.Status.ABORTED for sample in flattened):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    if any(sample.reward is None for sample in flattened):
        return DynamicFilterOutput(keep=False, reason="group_has_unscored_game")
    if any(_chess_outcome(sample) == "error" for sample in flattened):
        return DynamicFilterOutput(keep=False, reason="group_has_chess_infrastructure_error")
    return DynamicFilterOutput(keep=True)
