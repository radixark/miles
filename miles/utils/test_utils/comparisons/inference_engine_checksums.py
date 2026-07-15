from pathlib import Path

from miles.utils.audit_utils.event_analyzer.rules import inference_engine_weight_checksum_consistency
from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


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

    baseline_by_rollout = _checksums_by_rollout_id(baseline)
    target_by_rollout = _checksums_by_rollout_id(target)
    assert baseline_by_rollout.keys() == target_by_rollout.keys(), (
        f"Engine checksum rollout_id sets differ: "
        f"baseline={sorted(baseline_by_rollout)} "
        f"vs target={sorted(target_by_rollout)}"
    )

    mismatches: list[ChecksumMismatchIssue] = []
    for rollout_id in sorted(baseline_by_rollout):
        mismatches += list(
            compare_flat_dicts(
                a=baseline_by_rollout[rollout_id],
                b=target_by_rollout[rollout_id],
                label_a=f"baseline/rollout_{rollout_id}",
                label_b=f"target/rollout_{rollout_id}",
            )
        )
    assert not mismatches, "Engine weight checksum baseline-vs-target mismatch:\n" + "\n".join(
        f"  - {m.label_a} vs {m.label_b} key {m.key}: {m.value_a} != {m.value_b}" for m in mismatches
    )
    print(f"Engine weight checksum comparison passed: {len(baseline_by_rollout)} rollout(s) compared")


def _checksums_by_rollout_id(events: list[InferenceEngineWeightChecksumEvent]) -> dict[int, dict[str, str]]:
    by_rollout: dict[int, dict[str, str]] = {}
    for event in events:
        if event.rollout_id is None:
            continue
        assert (
            event.rollout_id not in by_rollout
        ), f"Duplicate InferenceEngineWeightChecksumEvent for rollout {event.rollout_id}"
        assert event.engine_checksums, f"No engine checksums for rollout {event.rollout_id}"
        by_rollout[event.rollout_id] = event.engine_checksums[0]
    return by_rollout


def _read_inference_engine_checksum_events(dump_dir: Path) -> list[InferenceEngineWeightChecksumEvent]:
    """Read all InferenceEngineWeightChecksumEvents from the events directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, InferenceEngineWeightChecksumEvent)]
