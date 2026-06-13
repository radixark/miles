"""Tests for event_analyzer rules/engine_weight_checksum_consistency."""

from datetime import datetime, timezone

from miles.utils.event_analyzer.rules.engine_weight_checksum_consistency import check
from miles.utils.event_logger.models import EngineWeightChecksumEvent
from miles.utils.process_identity import MainProcessIdentity

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_event(*, rollout_id: int, engine_index: int, checksums: dict[str, str]) -> EngineWeightChecksumEvent:
    return EngineWeightChecksumEvent(
        timestamp=_FIXED_TS,
        source=MainProcessIdentity(),
        rollout_id=rollout_id,
        engine_index=engine_index,
        checksums=checksums,
    )


class TestCheck:
    def test_empty_events_no_mismatches(self) -> None:
        """No engine checksum events means nothing to check."""
        assert check([]) == []

    def test_single_engine_no_comparison(self) -> None:
        """A single engine has no peer to compare against."""
        events = [_make_event(rollout_id=0, engine_index=0, checksums={"rank0/w": "aaa"})]
        assert check(events) == []

    def test_matching_engines_no_mismatches(self) -> None:
        """All engines holding identical checksums produce no issues."""
        events = [
            _make_event(rollout_id=0, engine_index=0, checksums={"rank0/w": "aaa"}),
            _make_event(rollout_id=0, engine_index=1, checksums={"rank0/w": "aaa"}),
            _make_event(rollout_id=0, engine_index=2, checksums={"rank0/w": "aaa"}),
        ]
        assert check(events) == []

    def test_tensor_mismatch_reports_engine_and_tensor(self) -> None:
        """A single differing tensor on one engine is reported with engine and tensor labels."""
        events = [
            _make_event(rollout_id=5, engine_index=0, checksums={"rank0/w": "aaa"}),
            _make_event(rollout_id=5, engine_index=1, checksums={"rank0/w": "zzz"}),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].key == "rank0/w"
        assert mismatches[0].label_a == "rollout_5/engine_0"
        assert mismatches[0].label_b == "rollout_5/engine_1"

    def test_missing_tensor_on_one_engine_detected(self) -> None:
        """A tensor present on engine 0 but absent on engine 1 is a mismatch."""
        events = [
            _make_event(rollout_id=0, engine_index=0, checksums={"rank0/w": "aaa", "rank0/b": "bbb"}),
            _make_event(rollout_id=0, engine_index=1, checksums={"rank0/w": "aaa"}),
        ]
        mismatches = check(events)
        assert any(m.key == "rank0/b" and "<missing>" in m.value_b for m in mismatches)

    def test_engines_compared_against_lowest_index(self) -> None:
        """Out-of-order engine_index events still compare every engine against index 0."""
        events = [
            _make_event(rollout_id=0, engine_index=2, checksums={"rank0/w": "bbb"}),
            _make_event(rollout_id=0, engine_index=0, checksums={"rank0/w": "aaa"}),
            _make_event(rollout_id=0, engine_index=1, checksums={"rank0/w": "aaa"}),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].label_a == "rollout_0/engine_0"
        assert mismatches[0].label_b == "rollout_0/engine_2"

    def test_only_mismatched_rollout_reported(self) -> None:
        """Engines are grouped per rollout; only the inconsistent rollout yields issues."""
        events = [
            _make_event(rollout_id=0, engine_index=0, checksums={"rank0/w": "aaa"}),
            _make_event(rollout_id=0, engine_index=1, checksums={"rank0/w": "aaa"}),
            _make_event(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"}),
            _make_event(rollout_id=1, engine_index=1, checksums={"rank0/w": "zzz"}),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert "rollout_1/" in mismatches[0].label_a
