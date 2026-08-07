import pytest

from miles.utils.workers import colocate_matching
from miles.utils.workers.colocate_matching import GpuPlacement, assert_same_nodes, local_gpu_index, match_by_gpu


def _placement(worker_name, node_name="gpu-1", gpu_ids=(0,)):
    return GpuPlacement(worker_name=worker_name, node_name=node_name, gpu_ids=tuple(gpu_ids))


class TestMatchByGpu:
    def test_pairs_a_whole_node_engine_with_the_trainers_on_it(self):
        """Eight trainer ranks and one engine holding the same eight gpus, in engine rank order."""
        engine = [_placement("engine-0-0", gpu_ids=range(8))]
        trainers = [_placement(f"trainer-0-{index}", gpu_ids=(index,)) for index in range(8)]

        pairing = match_by_gpu(engine_placements=engine, trainer_placements=trainers)

        assert pairing.trainer_names_in_engine_order() == [f"trainer-0-{index}" for index in range(8)]

    def test_keeps_engine_rank_order_rather_than_gpu_order(self):
        """The transfer group is indexed by engine rank, so the order is part of the answer."""
        engine = [_placement("engine-0-0", gpu_ids=(3, 1))]
        trainers = [_placement("t1", gpu_ids=(1,)), _placement("t3", gpu_ids=(3,))]

        pairing = match_by_gpu(engine_placements=engine, trainer_placements=trainers)

        assert pairing.trainer_names_in_engine_order() == ["t3", "t1"]

    def test_distinguishes_the_same_gpu_index_on_two_nodes(self):
        """gpu 0 of two machines are different cards, and confusing them would pair across the network."""
        engine = [
            _placement("e-a", node_name="gpu-1", gpu_ids=(0,)),
            _placement("e-b", node_name="gpu-2", gpu_ids=(0,)),
        ]
        trainers = [
            _placement("t-a", node_name="gpu-1", gpu_ids=(0,)),
            _placement("t-b", node_name="gpu-2", gpu_ids=(0,)),
        ]

        pairing = match_by_gpu(engine_placements=engine, trainer_placements=trainers)

        assert pairing.trainer_names_in_engine_order() == ["t-a", "t-b"]

    def test_refuses_an_engine_gpu_no_trainer_holds(self):
        """A weight update to an unshared gpu transfers nothing, and training would learn from stale weights."""
        engine = [_placement("engine-0-0", gpu_ids=(0, 1))]
        trainers = [_placement("trainer-0-0", gpu_ids=(0,))]

        with pytest.raises(AssertionError, match="no trainer shares"):
            match_by_gpu(engine_placements=engine, trainer_placements=trainers)

    def test_refuses_two_trainers_on_one_gpu(self):
        """That cannot happen on a real cluster, so the observation is wrong and must not be built on."""
        engine = [_placement("engine-0-0", gpu_ids=(0,))]
        trainers = [_placement("t-a", gpu_ids=(0,)), _placement("t-b", gpu_ids=(0,))]

        with pytest.raises(AssertionError, match="claimed by both"):
            match_by_gpu(engine_placements=engine, trainer_placements=trainers)

    def test_refuses_two_engine_ranks_on_one_gpu(self):
        """Both would receive the same weights and one of them would be serving the wrong shard."""
        engine = [_placement("e-a", gpu_ids=(0,)), _placement("e-b", gpu_ids=(0,))]
        trainers = [_placement("t", gpu_ids=(0,))]

        with pytest.raises(AssertionError, match="more than one engine rank"):
            match_by_gpu(engine_placements=engine, trainer_placements=trainers)

    def test_allows_trainers_the_engine_does_not_cover(self):
        """A trainer cell wider than its engine is the ordinary case, not an error."""
        engine = [_placement("engine-0-0", gpu_ids=(0,))]
        trainers = [_placement("t0", gpu_ids=(0,)), _placement("t1", gpu_ids=(1,))]

        assert match_by_gpu(engine_placements=engine, trainer_placements=trainers).trainer_names_in_engine_order() == [
            "t0"
        ]


class TestAssertSameNodes:
    def test_accepts_a_correctly_placed_group(self):
        """The pairing controller's job, verified independently before any transfer is set up."""
        assert_same_nodes(
            engine_placements=[_placement("e", node_name="gpu-1")],
            trainer_placements=[_placement("t", node_name="gpu-1")],
        )

    def test_refuses_an_engine_that_landed_elsewhere(self):
        """Failing at setup beats failing halfway through a weight update."""
        with pytest.raises(AssertionError, match="engines run on"):
            assert_same_nodes(
                engine_placements=[_placement("e", node_name="gpu-9")],
                trainer_placements=[_placement("t", node_name="gpu-1")],
            )


class TestLocalGpuIndex:
    def test_translates_a_card_into_this_process_own_numbering(self):
        """A handle carries the producer's device index, and the consumer may see a different set."""
        assert local_gpu_index(gpu_uuid="GPU-c", visible_uuids=["GPU-a", "GPU-b", "GPU-c"]) == 2

    def test_refuses_a_card_this_process_cannot_see(self):
        """Guessing an index would open a handle onto someone else's memory."""
        with pytest.raises(AssertionError, match="not visible here"):
            local_gpu_index(gpu_uuid="GPU-z", visible_uuids=["GPU-a"])


class TestLayoutPairs:
    def test_allows_trainer_cells_that_seat_no_engine(self):
        """A prefill pool_id on its own nodes leaves trainer gpus with no engine, which is a legal run."""
        layout = colocate_matching.PairingLayout(
            engine_cells=2, trainer_cells=4, pods_per_engine_cell=1, pods_per_trainer_cell=1
        )

        colocate_matching.assert_layout_pairs(layout=layout)

    def test_allows_a_trainer_cell_seating_fewer_engines_than_it_could(self):
        """Half a trainer cell may run engines and the other half none, which is still rank-for-rank."""
        layout = colocate_matching.PairingLayout(
            engine_cells=2, trainer_cells=1, pods_per_engine_cell=1, pods_per_trainer_cell=4
        )

        colocate_matching.assert_layout_pairs(layout=layout)

    def test_refuses_more_engine_cells_than_the_trainer_cells_seat(self):
        """The surplus engines would pair with a trainer cell the run never created."""
        layout = colocate_matching.PairingLayout(
            engine_cells=8, trainer_cells=1, pods_per_engine_cell=1, pods_per_trainer_cell=4
        )

        with pytest.raises(AssertionError, match="do not fit"):
            colocate_matching.assert_layout_pairs(layout=layout)

    def test_refuses_an_engine_cell_that_straddles_two_trainer_cells(self):
        """No single trainer cell then owns the engine, so healing one would leave the other half live."""
        layout = colocate_matching.PairingLayout(
            engine_cells=1, trainer_cells=1, pods_per_engine_cell=2, pods_per_trainer_cell=3
        )

        with pytest.raises(AssertionError, match="whole number"):
            colocate_matching.assert_layout_pairs(layout=layout)
