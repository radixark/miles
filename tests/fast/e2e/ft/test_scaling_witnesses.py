from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import scenario_inference_scaling as inference
from tests.e2e.ft.conftest_ft import scenario_trainer_scaling as trainer
from tests.fast.utils.soak.soak_fakes import _at, _write_sut_lines

from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


def _train_event(rollout_id: int, outcomes: dict[int, str | list[TrainStepOutcome]]) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(
        timestamp=_at(rollout_id + 10),
        source=TrainerControllerProcessIdentity(trainer_id="actor"),
        rollout_id=rollout_id,
        attempt=0,
        role="actor",
        cell_outcomes=outcomes,
    )


def _engine_event(rollout_id: int, count: int) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=_at(rollout_id + 10),
        source=TrainerControllerProcessIdentity(trainer_id="actor"),
        rollout_id=rollout_id,
        weight_version=rollout_id + 1,
        debug_trainer_load_state_timestamp=0,
        debug_weight_update_id=str(rollout_id),
        engine_snapshots=[
            dict(cell_id=f"engine-{i}", workers_hash=f"hash-{i}", tensor_checksums={"w": "x"}) for i in range(count)
        ],
    )


class TestEngineScalingWitness:
    def test_weight_updates_are_attributed_to_the_next_rollout_and_final_sync_is_excluded(
        self, tmp_path: Path
    ) -> None:
        """Startup weights witness rollout zero while the final unused update does not add a rollout."""
        (tmp_path / "events").mkdir()
        _write_sut_lines(
            tmp_path / "events/controller.jsonl", [_engine_event(-1, 2), _engine_event(1, 3), _engine_event(11, 2)]
        )
        assert inference._read_engine_counts_of_rollout_id([], tmp_path) == {0: 2, 2: 3}

    def test_duplicate_weight_updates_cannot_overwrite_a_rollout_witness(self, tmp_path: Path) -> None:
        """Two updates for one rollout fail instead of hiding inconsistent engine counts."""
        (tmp_path / "events").mkdir()
        _write_sut_lines(tmp_path / "events/controller.jsonl", [_engine_event(1, 2), _engine_event(1, 3)])
        with pytest.raises(AssertionError, match="two weight updates"):
            inference._read_engine_counts_of_rollout_id([], tmp_path)


class TestTrainerScalingWitness:
    @pytest.mark.parametrize(
        "rollout_id,cell_index,accepted", [(7, 2, True), (10, 2, True), (6, 2, False), (11, 2, False), (7, 0, False)]
    )
    def test_only_the_removed_cell_can_fail_inside_the_inclusive_shrink_landing_window(
        self, tmp_path: Path, rollout_id: int, cell_index: int, accepted: bool
    ) -> None:
        """Shrink tolerance never permits a surviving cell or a failure outside the landing window."""
        (tmp_path / "events").mkdir()
        event = _train_event(rollout_id, {cell_index: "error", 1: [TrainStepOutcome.NORMAL]})
        _write_sut_lines(tmp_path / "events/controller.jsonl", [event])
        if accepted:
            assert trainer._read_num_trained_cells_of_rollout_id([], tmp_path) == {rollout_id: 2}
        else:
            with pytest.raises(AssertionError, match="only the"):
                trainer._read_num_trained_cells_of_rollout_id([], tmp_path)

    def test_duplicate_training_steps_are_rejected(self, tmp_path: Path) -> None:
        """Multiple final attempts cannot masquerade as one completed rollout."""
        (tmp_path / "events").mkdir()
        event = _train_event(0, {0: [TrainStepOutcome.NORMAL]})
        _write_sut_lines(tmp_path / "events/controller.jsonl", [event, event])
        with pytest.raises(AssertionError, match="ends rollout 0 twice"):
            trainer._read_num_trained_cells_of_rollout_id([], tmp_path)

    @pytest.mark.parametrize("outcome", [TrainStepOutcome.NORMAL, TrainStepOutcome.DISCARDED_SHOULD_RETRY])
    def test_completed_cells_must_all_report_normal(self, tmp_path: Path, outcome: TrainStepOutcome) -> None:
        """A discarded training attempt is not successful evidence even inside a shrink window."""
        (tmp_path / "events").mkdir()
        _write_sut_lines(tmp_path / "events/controller.jsonl", [_train_event(7, {0: [outcome]})])
        if outcome is TrainStepOutcome.NORMAL:
            assert trainer._read_num_trained_cells_of_rollout_id([], tmp_path) == {7: 1}
        else:
            with pytest.raises(AssertionError, match="only the"):
                trainer._read_num_trained_cells_of_rollout_id([], tmp_path)
