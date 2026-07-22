"""Tests for test_utils.comparisons.inference_engine_checksums.compare_inference_engine_checksums."""

from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.test_utils.comparisons.inference_engine_checksums import compare_inference_engine_checksums


def _write_inference_engine_events(side_dir: Path, partials: list[dict[str, Any]]) -> None:
    events_dir = side_dir / "events"
    event_logger = EventLogger(log_dir=events_dir, source=MainProcessIdentity())
    for partial in partials:
        event_logger.log(InferenceEngineWeightChecksumEvent, partial, print_log=False)
    event_logger.close()


def _partial(*, rollout_id: int | None, engine_checksums: list[dict[str, str]]) -> dict[str, Any]:
    return dict(rollout_id=rollout_id, engine_checksums=engine_checksums)


class TestCompareInferenceEngineChecksums:
    def test_identical_passes(self, tmp_path: Path) -> None:
        """Internally-consistent sides with equal representative checksums pass."""
        partials = [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}])]
        _write_inference_engine_events(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_differing_engine_counts_still_pass(self, tmp_path: Path) -> None:
        """Engine count may differ between sides; only internal agreement + representative equality matter."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}, {"rank0/w": "aaa"}])],
        )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_none_rollout_id_skipped(self, tmp_path: Path) -> None:
        """The initial out-of-loop sync (rollout_id=None) is not compared: it differs here yet the
        per-rollout checksums match, so the comparison still passes."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=None, engine_checksums=[{"rank0/w": "init_baseline"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=None, engine_checksums=[{"rank0/w": "init_target"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
            ],
        )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_recurring_none_across_phases_skipped(self, tmp_path: Path) -> None:
        """A multi-phase resume yields several None events per side; all are skipped, so a side with
        more None events than the other still passes when the per-rollout checksums match."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=None, engine_checksums=[{"rank0/w": "init_a"}]),
                _partial(rollout_id=2, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=None, engine_checksums=[{"rank0/w": "init_b"}]),
                _partial(rollout_id=5, engine_checksums=[{"rank0/w": "bbb"}]),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=2, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=5, engine_checksums=[{"rank0/w": "bbb"}]),
            ],
        )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_baseline_engines_disagree_fails(self, tmp_path: Path) -> None:
        """If baseline's own engines disagree, the comparison fails (caught by the consistency rule)."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match="Baseline engines disagree"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_target_engines_disagree_fails(self, tmp_path: Path) -> None:
        """If target's own engines disagree, the comparison fails."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        )

        with pytest.raises(AssertionError, match="Target engines disagree"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_representative_mismatch_fails(self, tmp_path: Path) -> None:
        """Internally-consistent sides whose representatives differ fail and name the tensor."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "zzz"}])]
        )

        with pytest.raises(AssertionError, match=r"key rank0/w"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_missing_rollout_fails(self, tmp_path: Path) -> None:
        """A rollout present only on one side fails closed."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=2, engine_checksums=[{"rank0/w": "ccc"}]),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match="rollout_id sets differ"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_empty_baseline_fails(self, tmp_path: Path) -> None:
        """No baseline events fails closed rather than vacuously passing."""
        _write_inference_engine_events(tmp_path / "baseline", [])
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match="No InferenceEngineWeightChecksumEvents found in baseline"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))
