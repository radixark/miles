from collections.abc import Iterable

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent

__all__ = ["check"]


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Check: all engines of one rollout must hold exactly the same weights."""
    issues: list[ChecksumMismatchIssue] = []
    seen: set[tuple[str | None, str, int, str, str]] = set()
    representatives: dict[tuple[str | None, str, int], tuple[str, dict[str, str]]] = {}
    for event in events:
        if isinstance(event, InferenceEngineWeightChecksumEvent):
            if event.engine_snapshots:
                assert event.weight_version is not None
                for snapshot in event.engine_snapshots:
                    group = (event.trainer_model_id, snapshot.model_name, event.weight_version)
                    identity = (*group, snapshot.cell_id, snapshot.workers_hash)
                    assert identity not in seen, f"Duplicate engine checksum evidence: {identity}"
                    seen.add(identity)
                    label = f"{group}/cell_{snapshot.cell_id}/{snapshot.workers_hash}"
                    if group in representatives:
                        previous_label, tensors = representatives[group]
                        issues += list(
                            compare_flat_dicts(a=tensors, b=snapshot.tensors, label_a=previous_label, label_b=label)
                        )
                    else:
                        representatives[group] = (label, snapshot.tensors)
                continue
            issues += list(_check_one_rollout(event))
    return issues


def _check_one_rollout(event: InferenceEngineWeightChecksumEvent) -> Iterable[ChecksumMismatchIssue]:
    engines = event.engine_checksums
    if len(engines) < 2:
        return
    baseline = engines[0]
    for engine_index in range(1, len(engines)):
        yield from compare_flat_dicts(
            a=baseline,
            b=engines[engine_index],
            label_a=f"rollout_{event.rollout_id}/engine_0",
            label_b=f"rollout_{event.rollout_id}/engine_{engine_index}",
        )
