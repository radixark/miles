from dataclasses import dataclass
from datetime import datetime

from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import IssuedSampleIdentityIssue
from miles.utils.audit_utils.event_logger.models import DataSourceIssuedSamplesEvent


@dataclass(frozen=True)
class _IssuedSample:
    group_index: int | None
    slot: int | None
    sample_index: int
    issued_at: datetime | None


def _issued_samples(
    events: list[DataSourceIssuedSamplesEvent],
) -> tuple[list[_IssuedSample], list[IssuedSampleIdentityIssue]]:
    by_identity: dict[tuple[int, int, int], _IssuedSample] = {}
    identities_by_sample: dict[int, set[tuple[int, int]]] = {}

    for event in events:
        for group in event.groups:
            for slot, sample_index in enumerate(group.sample_indices):
                key = (group.group_index, slot, sample_index)
                sample = _IssuedSample(
                    group_index=group.group_index,
                    slot=slot,
                    sample_index=sample_index,
                    issued_at=event.timestamp,
                )
                if key not in by_identity or event.timestamp < by_identity[key].issued_at:
                    by_identity[key] = sample
                identities_by_sample.setdefault(sample_index, set()).add((group.group_index, slot))

    conflicted = {sample_index for sample_index, identities in identities_by_sample.items() if len(identities) > 1}
    issues = [
        IssuedSampleIdentityIssue(
            description="the same sample index was issued for multiple GRPO slots",
            sample_index=sample_index,
            identities=[f"group {group_index} slot {slot}" for group_index, slot in sorted(identities)],
        )
        for sample_index, identities in sorted(identities_by_sample.items())
        if sample_index in conflicted
    ]
    samples = [sample for key, sample in sorted(by_identity.items()) if sample.sample_index not in conflicted]
    return samples, issues
