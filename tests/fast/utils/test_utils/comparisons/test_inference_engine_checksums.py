"""Tests for test_utils.comparisons.inference_engine_checksums.compare_inference_engine_checksums."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainerControllerProcessIdentity
from miles.utils.test_utils.comparisons.inference_engine_checksums import (
    assert_engine_count,
    assert_engine_weights_moved,
    compare_inference_engine_checksums,
)


def _write_inference_engine_events(
    side_dir: Path, partials: list[dict[str, Any]], *, model_id: str | None = None
) -> None:
    events_dir = side_dir / EVENTS_DIRNAME
    source = _source_of(model_id)
    event_logger = EventLogger(log_dir=events_dir, source=source, file_name=f"{source.to_name()}.jsonl")
    for partial in partials:
        event_logger.log(InferenceEngineWeightChecksumEvent, partial, print_log=False)
    event_logger.close()


def _write_events_without_trainer_model_id(side_dir: Path, partials: list[dict[str, Any]]) -> None:
    events_dir = side_dir / EVENTS_DIRNAME
    events_dir.mkdir(parents=True, exist_ok=True)
    source = _source_of(None)
    lines: list[str] = []
    for partial in partials:
        dumped = InferenceEngineWeightChecksumEvent(
            **partial, timestamp=datetime.now(timezone.utc), source=source
        ).model_dump(mode="json")
        del dumped["trainer_model_id"]
        lines.append(json.dumps(dumped))
    (events_dir / f"{source.to_name()}.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _source_of(model_id: str | None) -> SimpleProcessIdentity | TrainerControllerProcessIdentity:
    if model_id is None:
        return SimpleProcessIdentity(component="main")
    return TrainerControllerProcessIdentity(trainer_id=f"{model_id}-actor", model_id=model_id)


def _partial(
    *, rollout_id: int, engine_checksums: list[dict[str, str]], trainer_model_id: str | None = None
) -> dict[str, Any]:
    return dict(
        rollout_id=rollout_id,
        trainer_model_id=trainer_model_id,
        weight_version=rollout_id + 1,
        debug_trainer_load_state_timestamp=0.0,
        debug_weight_update_id=f"update-{rollout_id + 1}",
        engine_snapshots=[
            dict(cell_id=f"cell-{index}", workers_hash=f"incarnation-{index}", tensor_checksums=checksums)
            for index, checksums in enumerate(engine_checksums)
        ],
    )


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

    def test_the_fresh_startup_sync_is_compared_like_any_other_rollout(self, tmp_path: Path) -> None:
        """The startup push is stamped -1 rather than skipped, so two runs that boot from different weights fail."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "init_baseline"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "init_target"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
            ],
        )

        with pytest.raises(AssertionError):
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

        with pytest.raises(AssertionError, match=r"\(model_id, weight_version\) sets differ"):
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
                tmp_path / side,
                [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a")],
                model_id="a",
            )
            _write_inference_engine_events(
                tmp_path / side,
                [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}], trainer_model_id="b")],
                model_id="b",
            )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_policy_whose_weights_differ_is_reported(self, tmp_path: Path) -> None:
        """Comparing only one of the two policies would hide exactly the drift this comparison exists to catch."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a")],
            model_id="a",
        )
        _write_inference_engine_events(
            tmp_path / "baseline",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}], trainer_model_id="b")],
            model_id="b",
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a")],
            model_id="a",
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "ccc"}], trainer_model_id="b")],
            model_id="b",
        )

        with pytest.raises(AssertionError, match=r"baseline/b/version_2 vs target/b/version_2"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_the_writers_identity_no_longer_decides_which_policy_an_event_belongs_to(self, tmp_path: Path) -> None:
        """The orchestration script writes every policy's event, so only the payload can name the policy."""
        for side in ("baseline", "target"):
            _write_inference_engine_events(
                tmp_path / side,
                [
                    _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a"),
                    _partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}], trainer_model_id="b"),
                ],
            )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestPublishedVersionKeys:
    @staticmethod
    def _at(*, rollout_id: int, weight_version: int, checksum: str) -> dict[str, Any]:
        return {
            **_partial(rollout_id=rollout_id, engine_checksums=[{"rank0/w": checksum}]),
            "weight_version": weight_version,
        }

    def test_sides_whose_rollout_ids_differ_align_on_the_published_version(self, tmp_path: Path) -> None:
        """A retried rollout publishes one version under another rollout id, which must still compare."""
        _write_inference_engine_events(tmp_path / "baseline", [self._at(rollout_id=1, weight_version=2, checksum="a")])
        _write_inference_engine_events(tmp_path / "target", [self._at(rollout_id=4, weight_version=2, checksum="a")])

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_one_rollout_id_with_different_versions_is_not_merged(self, tmp_path: Path) -> None:
        """Keying by rollout id would pair version 2 against version 3 and hide the missing publication."""
        _write_inference_engine_events(tmp_path / "baseline", [self._at(rollout_id=1, weight_version=2, checksum="a")])
        _write_inference_engine_events(tmp_path / "target", [self._at(rollout_id=1, weight_version=3, checksum="a")])

        with pytest.raises(AssertionError, match=r"sets differ"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_repeated_version_with_the_same_weights_is_accepted(self, tmp_path: Path) -> None:
        """Re-sampling one publication twice records the same weights and is not a conflict."""
        repeated = [
            self._at(rollout_id=1, weight_version=2, checksum="a"),
            self._at(rollout_id=2, weight_version=2, checksum="a"),
        ]
        _write_inference_engine_events(tmp_path / "baseline", repeated)
        _write_inference_engine_events(tmp_path / "target", repeated[:1])

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_repeated_version_with_different_weights_is_rejected(self, tmp_path: Path) -> None:
        """Two different weight sets under one published version cannot both be the version the trainer made."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                self._at(rollout_id=1, weight_version=2, checksum="a"),
                self._at(rollout_id=2, weight_version=2, checksum="b"),
            ],
        )
        _write_inference_engine_events(tmp_path / "target", [self._at(rollout_id=1, weight_version=2, checksum="a")])

        with pytest.raises(AssertionError, match=r"Conflicting checksums for \(None, 2\)"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestLogsPredatingTheTrainerModelIdField:
    def test_events_without_the_field_read_back_as_one_unnamed_policy(self, tmp_path: Path) -> None:
        """A log written before the event named its policy is a single policy run, so it compares under None."""
        partials = [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        _write_events_without_trainer_model_id(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_field_less_event_still_compares_its_checksums(self, tmp_path: Path) -> None:
        """Reading the missing field as None must not turn the old side into an unchecked one."""
        _write_events_without_trainer_model_id(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "zzz"}])]
        )

        with pytest.raises(AssertionError, match=r"key rank0/w"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestAssertEngineWeightsMoved:
    def test_fewer_than_two_updates_do_not_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        """A lone weight update provides no pair from which movement can be established."""
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match="there is no pair to compare"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))

    def test_identical_updates_do_not_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        """Reordered checksum items remain the same weights and cannot establish movement."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=0, engine_checksums=[{"rank0/a": "aaa", "rank0/b": "bbb"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/b": "bbb", "rank0/a": "aaa"}]),
            ],
        )

        with pytest.raises(AssertionError, match="byte-identical engine weights"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))

    def test_distinct_updates_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        """Two distinct checksum dictionaries establish that engine weights changed."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}]),
            ],
        )

        assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))


class TestAssertEngineCount:
    def test_a_run_every_update_of_which_covered_both_engines_passes(self, tmp_path: Path) -> None:
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]),
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "bbb"}, {"rank0/w": "bbb"}]),
            ],
        )

        assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_an_update_that_reached_one_engine_of_two_is_caught(self, tmp_path: Path) -> None:
        """An engine that dropped out mid-run still leaves a run that trains, so nothing else notices."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}]),
            ],
        )

        with pytest.raises(AssertionError, match=r"pushed to \[1, 2\] engine"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_a_startup_sync_that_reached_one_engine_of_two_is_caught(self, tmp_path: Path) -> None:
        """The engines are synced before the first rollout, and one that joined late missed those weights."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "bbb"}, {"rank0/w": "bbb"}]),
            ],
        )

        with pytest.raises(AssertionError, match=r"pushed to \[1, 2\] engine"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_a_side_that_pushed_weights_to_no_engine_at_all_is_caught(self, tmp_path: Path) -> None:
        """A side whose engines never registered would otherwise report every count it was asked for."""
        (tmp_path / "target" / EVENTS_DIRNAME).mkdir(parents=True)

        with pytest.raises(AssertionError, match="no engine ever took weights"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)
