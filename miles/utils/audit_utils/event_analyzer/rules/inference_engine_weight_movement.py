from itertools import pairwise

from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent


def check(events: list[Event]) -> list[str]:
    versions: dict[tuple[str | None, str, str | None], dict[int, dict[str, str]]] = {}
    policies: dict[tuple[str | None, str, str | None], tuple[str, ...]] = {}
    for event in events:
        if not isinstance(event, InferenceEngineWeightChecksumEvent) or event.weight_version is None:
            continue
        assert event.movement_skip_reasons is not None, "Versioned checksum lacks movement applicability"
        for snapshot in event.engine_snapshots:
            model = (event.trainer_model_id, snapshot.model_name, event.version_epoch)
            policy = tuple(sorted(event.movement_skip_reasons))
            if model in policies:
                assert policies[model] == policy, f"Checksum movement applicability changed for {model}"
            policies[model] = policy
            if policy:
                continue
            by_version = versions.setdefault(model, {})
            if event.weight_version in by_version:
                assert (
                    by_version[event.weight_version] == snapshot.tensors
                ), f"Same-version weights disagree for {model}"
            else:
                by_version[event.weight_version] = snapshot.tensors

    issues: list[str] = []
    for model, by_version in versions.items():
        ordered = sorted(by_version)
        for before, after in pairwise(ordered):
            previous, current = by_version[before], by_version[after]
            if previous.keys() != current.keys():
                issues.append(f"Checksum tensor set changed for {model} between versions {before} and {after}")
            elif previous == current:
                issues.append(f"Weights did not change for {model} between versions {before} and {after}")
    return issues
