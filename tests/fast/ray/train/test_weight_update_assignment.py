import pytest

from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.train.weight_update_assignment import split_update_targets


def _info(count: int) -> UpdatableEngines:
    return UpdatableEngines(
        rollout_engines=[f"client-{index}" for index in range(count)],
        engine_gpu_counts=[index + 1 for index in range(count)],
        engine_gpu_offsets=[index * 8 for index in range(count)],
        engine_cell_ids=[f"cell-{index}" for index in range(count)],
        snapshot_cell_id_to_hashes={f"cell-{index}": f"hash-{index}" for index in range(count)},
    )


class TestSplitUpdateTargets:
    """Every inference cell is written by exactly one trainer cell, or it silently keeps old weights."""

    def test_an_even_split_gives_each_trainer_the_same_number_of_targets(self):
        """An unbalanced split makes one trainer the straggler that every update waits for."""
        assignments = split_update_targets(_info(4), num_trainer_cells=2)

        assert [a.engine_cell_ids for a in assignments] == [["cell-0", "cell-1"], ["cell-2", "cell-3"]]

    def test_the_remainder_goes_to_the_earliest_trainers(self):
        """The leftover targets must land somewhere, and the order must be the same on every call."""
        assignments = split_update_targets(_info(5), num_trainer_cells=3)

        assert [a.engine_cell_ids for a in assignments] == [["cell-0", "cell-1"], ["cell-2", "cell-3"], ["cell-4"]]

    def test_every_target_is_assigned_exactly_once(self):
        """A duplicated target is written twice, and a dropped one keeps serving the previous weights."""
        assignments = split_update_targets(_info(7), num_trainer_cells=3)

        assigned = [cell_id for a in assignments for cell_id in a.engine_cell_ids]
        assert assigned == [f"cell-{index}" for index in range(7)]

    def test_more_trainers_than_targets_leaves_the_extra_trainers_empty(self):
        """An empty assignment must be representable so the controller can skip that trainer entirely."""
        assignments = split_update_targets(_info(1), num_trainer_cells=3)

        assert [a.engine_cell_ids for a in assignments] == [["cell-0"], [], []]

    def test_every_per_engine_list_is_sliced_together(self):
        """Slicing the ids alone hands a trainer another engine's client, GPU count and address."""
        [assignment, _rest] = split_update_targets(_info(4), num_trainer_cells=2)

        assert assignment.rollout_engines == ["client-0", "client-1"]
        assert assignment.engine_gpu_counts == [1, 2]
        assert assignment.engine_gpu_offsets == [0, 8]
        assert assignment.snapshot_cell_id_to_hashes == {"cell-0": "hash-0", "cell-1": "hash-1"}

    def test_each_assignment_carries_only_its_own_worker_generations(self):
        """A trainer that sees a foreign generation could report an outcome for a cell it never wrote to."""
        assignments = split_update_targets(_info(4), num_trainer_cells=2)

        for assignment in assignments:
            assert set(assignment.snapshot_cell_id_to_hashes) == set(assignment.engine_cell_ids)

    def test_an_empty_snapshot_gives_every_trainer_nothing_to_do(self):
        """With no updatable engine there is nothing to send, and no trainer should be woken for it."""
        assignments = split_update_targets(_info(0), num_trainer_cells=2)

        assert [a.engine_cell_ids for a in assignments] == [[], []]

    def test_splitting_across_no_trainer_is_rejected(self):
        """Dividing the targets by zero trainers would silently drop the whole update."""
        with pytest.raises(AssertionError, match="at least one trainer cell"):
            split_update_targets(_info(2), num_trainer_cells=0)
