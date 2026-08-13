"""Tests for test_utils.comparisons.inference_engine_checksums.compare_inference_engine_checksums."""

from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainerControllerProcessIdentity
from miles.utils.test_utils.comparisons.inference_engine_checksums import compare_inference_engine_checksums


def _write_inference_engine_events(
    side_dir: Path, partials: list[dict[str, Any]], *, model_id: str | None = None
) -> None:
    events_dir = side_dir / "events"
    source = (
        SimpleProcessIdentity(component="main")
        if model_id is None
        else TrainerControllerProcessIdentity(trainer_id=f"{model_id}-actor", model_id=model_id)
    )
    event_logger = EventLogger(log_dir=events_dir, source=source, file_name=f"{source.to_name()}.jsonl")
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


class TestSeveralPolicies:
    def test_the_same_rollout_id_of_two_policies_is_not_a_duplicate(self, tmp_path: Path) -> None:
        """Every policy counts its own rollouts, so keying by rollout id alone rejects a legal multi policy run."""
        for side in ("baseline", "target"):
            _write_inference_engine_events(
                tmp_path / side, [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])], model_id="a"
            )
            _write_inference_engine_events(
                tmp_path / side, [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}])], model_id="b"
            )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_policy_whose_weights_differ_is_reported(self, tmp_path: Path) -> None:
        """Comparing only one of the two policies would hide exactly the drift this comparison exists to catch."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])], model_id="a"
        )
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}])], model_id="b"
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])], model_id="a"
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "ccc"}])], model_id="b"
        )

        with pytest.raises(AssertionError, match=r"baseline/b/rollout_1 vs target/b/rollout_1"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))
