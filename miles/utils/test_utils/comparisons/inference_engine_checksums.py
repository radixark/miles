from pathlib import Path

from miles.utils.audit_utils.event_analyzer.rules import inference_engine_weight_checksum_consistency
from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


def compare_inference_engine_checksums(baseline_dir: str, target_dir: str) -> None:
    baseline = _read_inference_engine_checksum_events(Path(baseline_dir))
    target = _read_inference_engine_checksum_events(Path(target_dir))
    assert baseline, f"No InferenceEngineWeightChecksumEvents found in baseline dir: {baseline_dir}"
    assert target, f"No InferenceEngineWeightChecksumEvents found in target dir: {target_dir}"

    # Each side's engines must already agree internally (same invariant as the production rule), so
    # one representative engine per rollout then proves baseline == target regardless of engine count.
    assert not inference_engine_weight_checksum_consistency.check(
        baseline
    ), "Baseline engines disagree with each other"
    assert not inference_engine_weight_checksum_consistency.check(target), "Target engines disagree with each other"

    baseline_by_model_and_rollout = _checksums_by_model_and_rollout_id(baseline)
    target_by_model_and_rollout = _checksums_by_model_and_rollout_id(target)
    assert baseline_by_model_and_rollout.keys() == target_by_model_and_rollout.keys(), (
        f"Engine checksum (model_id, rollout_id) sets differ: "
        f"baseline={sorted(baseline_by_model_and_rollout)} "
        f"vs target={sorted(target_by_model_and_rollout)}"
    )

    mismatches: list[ChecksumMismatchIssue] = []
    for key in sorted(baseline_by_model_and_rollout):
        model_id, rollout_id = key
        mismatches += list(
            compare_flat_dicts(
                a=baseline_by_model_and_rollout[key],
                b=target_by_model_and_rollout[key],
                label_a=f"baseline/{model_id}/rollout_{rollout_id}",
                label_b=f"target/{model_id}/rollout_{rollout_id}",
            )
        )
    assert not mismatches, "Engine weight checksum baseline-vs-target mismatch:\n" + "\n".join(
        f"  - {m.label_a} vs {m.label_b} key {m.key}: {m.value_a} != {m.value_b}" for m in mismatches
    )
    print(f"Engine weight checksum comparison passed: {len(baseline_by_model_and_rollout)} rollout(s) compared")


def _checksums_by_model_and_rollout_id(
    events: list[InferenceEngineWeightChecksumEvent],
) -> dict[tuple[str | None, int], dict[str, str]]:
    by_model_and_rollout: dict[tuple[str | None, int], dict[str, str]] = {}
    for event in events:
        if event.rollout_id is None:
            continue
        key = (_compute_model_id(event), event.rollout_id)
        assert key not in by_model_and_rollout, f"Duplicate InferenceEngineWeightChecksumEvent for {key}"
        assert event.engine_checksums, f"No engine checksums for {key}"
        by_model_and_rollout[key] = event.engine_checksums[0]
    return by_model_and_rollout


def _compute_model_id(event: InferenceEngineWeightChecksumEvent) -> str | None:
    source = event.source
    return source.model_id if isinstance(source, TrainerControllerProcessIdentity) else None


def _read_inference_engine_checksum_events(dump_dir: Path) -> list[InferenceEngineWeightChecksumEvent]:
    """Read all InferenceEngineWeightChecksumEvents from the events directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, InferenceEngineWeightChecksumEvent)]
