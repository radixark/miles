"""Tests for event_analyzer rules/inference_engine_weight_checksum_consistency."""

from datetime import datetime, timezone

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_checksum_consistency import check
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_event(
    *, rollout_id: int | None, engine_checksums: list[dict[str, str]]
) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=_FIXED_TS,
        source=MainProcessIdentity(),
        rollout_id=rollout_id,
        engine_checksums=engine_checksums,
    )


class TestCheck:
    def test_empty_events_no_mismatches(self) -> None:
        """No engine checksum events means nothing to check."""
        assert check([]) == []

    def test_single_engine_no_comparison(self) -> None:
        """A single engine has no peer to compare against."""
        events = [_make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}])]
        assert check(events) == []

    def test_matching_engines_no_mismatches(self) -> None:
        """All engines holding identical checksums produce no issues."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}, {"rank0/w": "aaa"}])
        ]
        assert check(events) == []

    def test_tensor_mismatch_reports_engine_and_tensor(self) -> None:
        """A single differing tensor on one engine is reported with engine and tensor labels."""
        events = [_make_event(rollout_id=5, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].key == "rank0/w"
        assert mismatches[0].label_a == "rollout_5/engine_0"
        assert mismatches[0].label_b == "rollout_5/engine_1"

    def test_missing_tensor_on_one_engine_detected(self) -> None:
        """A tensor present on engine 0 but absent on engine 1 is a mismatch."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa", "rank0/b": "bbb"}, {"rank0/w": "aaa"}])
        ]
        mismatches = check(events)
        assert any(m.key == "rank0/b" and "<missing>" in m.value_b for m in mismatches)

    def test_all_engines_compared_against_first(self) -> None:
        """Every engine is compared against engine 0; a later engine's diff names engine_2."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}, {"rank0/w": "bbb"}])
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].label_a == "rollout_0/engine_0"
        assert mismatches[0].label_b == "rollout_0/engine_2"

    def test_none_rollout_id_mismatch_labelled_rollout_none(self) -> None:
        """The initial out-of-loop sync (rollout_id=None) still checks engines and labels them rollout_None."""
        events = [_make_event(rollout_id=None, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].label_a == "rollout_None/engine_0"
        assert mismatches[0].label_b == "rollout_None/engine_1"

    def test_only_mismatched_rollout_reported(self) -> None:
        """Each rollout is its own event; only the inconsistent rollout yields issues."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]),
            _make_event(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}]),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert "rollout_1/" in mismatches[0].label_a
