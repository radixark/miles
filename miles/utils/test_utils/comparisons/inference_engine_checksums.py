from pathlib import Path

from miles.utils.audit_utils.event_analyzer.rules import inference_engine_weight_checksum_consistency
from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


def compare_inference_engine_checksums(baseline_dir: str, target_dir: str) -> None:
    baseline = read_inference_engine_checksum_events(Path(baseline_dir))
    target = read_inference_engine_checksum_events(Path(target_dir))
    assert baseline, f"No InferenceEngineWeightChecksumEvents found in baseline dir: {baseline_dir}"
    assert target, f"No InferenceEngineWeightChecksumEvents found in target dir: {target_dir}"

    # Each side's engines must already agree internally (same invariant as the production rule), so
    # one representative engine per rollout then proves baseline == target regardless of engine count.
    assert not inference_engine_weight_checksum_consistency.check(
        baseline
    ), "Baseline engines disagree with each other"
    assert not inference_engine_weight_checksum_consistency.check(target), "Target engines disagree with each other"

    baseline_by_model_and_version = _checksums_by_model_and_version(baseline)
    target_by_model_and_version = _checksums_by_model_and_version(target)
    assert baseline_by_model_and_version.keys() == target_by_model_and_version.keys(), (
        f"Engine checksum (model_id, weight_version) sets differ: "
        f"baseline={sorted(baseline_by_model_and_version)} "
        f"vs target={sorted(target_by_model_and_version)}"
    )

    mismatches: list[ChecksumMismatchIssue] = []
    for key in sorted(baseline_by_model_and_version):
        model_id, weight_version = key
        mismatches += list(
            compare_flat_dicts(
                a=baseline_by_model_and_version[key],
                b=target_by_model_and_version[key],
                label_a=f"baseline/{model_id}/version_{weight_version}",
                label_b=f"target/{model_id}/version_{weight_version}",
            )
        )
    assert not mismatches, "Engine weight checksum baseline-vs-target mismatch:\n" + "\n".join(
        f"  - {m.label_a} vs {m.label_b} key {m.key}: {m.value_a} != {m.value_b}" for m in mismatches
    )
    print(f"Engine weight checksum comparison passed: {len(baseline_by_model_and_version)} version(s) compared")


def assert_engine_count(*, side: str, dump_dir: str, expected: int) -> None:
    events = read_inference_engine_checksum_events(Path(dump_dir))
    assert events, f"{side}: no InferenceEngineWeightChecksumEvents in {dump_dir}, so no engine ever took weights"

    counted = sorted({len(event.engine_snapshots) for event in events})
    assert counted == [expected], (
        f"{side}: weights were pushed to {counted} engine(s), not {expected}; a run served by fewer engines still "
        f"trains, so nothing else would notice engines that never joined"
    )

    print(f"{side}: every weight update covered {expected} engine(s)")


def assert_engine_weights_moved(*, side: str, dump_dir: str) -> None:
    by_model_and_version = _checksums_by_model_and_version(read_inference_engine_checksum_events(Path(dump_dir)))
    assert len(by_model_and_version) > 1, (
        f"{side}: engine weight checksums cover {sorted(by_model_and_version)}, so there is no pair to compare "
        f"and nothing proves the run pushed an update at all"
    )

    distinct: set[tuple[tuple[str, str], ...]] = {tuple(sorted(one.items())) for one in by_model_and_version.values()}
    assert len(distinct) > 1, (
        f"{side}: every one of {len(by_model_and_version)} versions pushed byte-identical engine weights, so the "
        f"optimizer moved nothing and a bitwise comparison against another such run would prove nothing"
    )

    print(
        f"{side}: engine weights moved across {len(by_model_and_version)} version(s), "
        f"{len(distinct)} distinct checksum(s)"
    )


def _checksums_by_model_and_version(
    events: list[InferenceEngineWeightChecksumEvent],
) -> dict[tuple[str | None, int], dict[str, str]]:
    by_model_and_version: dict[tuple[str | None, int], dict[str, str]] = {}
    for event in events:
        key = (event.trainer_model_id, event.weight_version)
        checksums = event.engine_snapshots[0].tensor_checksums
        assert by_model_and_version.setdefault(key, checksums) == checksums, f"Conflicting checksums for {key}"
    return by_model_and_version


def read_inference_engine_checksum_events(dump_dir: Path) -> list[InferenceEngineWeightChecksumEvent]:
    """Read all InferenceEngineWeightChecksumEvents from the events directory."""
    events_dir: Path = dump_dir / EVENTS_DIRNAME
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, InferenceEngineWeightChecksumEvent)]
