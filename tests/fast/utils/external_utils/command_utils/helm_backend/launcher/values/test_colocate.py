import pydantic
import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import LAYOUT, engine, trainer

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.workers.worker_spec import CommandWorkerSpec

COLOCATE_LAYOUT = LAYOUT.model_copy(update={"colocate": True})


def _disaggregated_engines(*, decode_offset: int) -> list[CommandWorkerSpec]:
    return [
        engine(num_cells=2, gpus_per_engine=8, name="inference-engine-0-0", gpu_offset=0),
        engine(num_cells=2, gpus_per_engine=8, name="inference-engine-0-1", gpu_offset=decode_offset),
    ]


class TestTheValuesCarryAWholePairingConfig:
    def test_reads_the_pools_that_start_inside_the_trainer_off_their_gpu_offsets(self):
        """The gpu offset is what already decides who shares the trainer's cards, so nothing else declares it."""
        specs = [*_disaggregated_engines(decode_offset=16), trainer(num_cells=4, gpus_per_cell=8)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert built["colocate"] == {
            "namespace": "rl",
            "release": "r",
            "trainer_pool_id": "trainer-engine-actor",
            "inference_pools": [
                {
                    "pool_id": "inference-engine-0-0",
                    "layout": {
                        "num_inference_cells": 2,
                        "num_trainer_cells": 4,
                        "num_pods_per_inference_cell": 1,
                        "num_pods_per_trainer_cell": 1,
                        "num_gpus_per_node": 8,
                        "num_gpus_per_inference_pod": 8,
                        "gpu_offset": 0,
                    },
                },
                {
                    "pool_id": "inference-engine-0-1",
                    "layout": {
                        "num_inference_cells": 2,
                        "num_trainer_cells": 4,
                        "num_pods_per_inference_cell": 1,
                        "num_pods_per_trainer_cell": 1,
                        "num_gpus_per_node": 8,
                        "num_gpus_per_inference_pod": 8,
                        "gpu_offset": 16,
                    },
                },
            ],
        }

    def test_names_the_pools_by_the_pool_id_the_pods_are_labelled_with(self):
        """The controller identifies a pod by its miles pool label, so the config has to speak that value."""
        specs = [*_disaggregated_engines(decode_offset=16), trainer(num_cells=4, gpus_per_cell=8)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert built["colocate"]["trainer_pool_id"] == built["trainerEngines"][0]["poolId"]
        assert [pool["pool_id"] for pool in built["colocate"]["inference_pools"]] == [
            entry["poolId"] for entry in built["inferenceEngines"]
        ]

    def test_leaves_out_a_pool_placed_past_the_trainer_gpus(self):
        """A prefill pool_id on its own nodes needs no pairing, and gating it would strand it forever."""
        specs = [*_disaggregated_engines(decode_offset=32), trainer(num_cells=2, gpus_per_cell=8)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert [pool["pool_id"] for pool in built["colocate"]["inference_pools"]] == ["inference-engine-0-0"]

    def test_pairs_a_pool_that_leaves_trainer_gpus_to_themselves(self):
        """Half the trainer may run no engine at all, which is still a rank-for-rank pairing where it does."""
        specs = [engine(num_cells=2, gpus_per_engine=8), trainer(num_cells=1, gpus_per_cell=32)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert [pool["pool_id"] for pool in built["colocate"]["inference_pools"]] == ["inference-engine-0-0"]

    def test_refuses_two_pools_whose_gpu_ranges_overlap(self):
        """Both would be pinned onto the same node, and one of them would find no gpus left to run on."""
        specs = [*_disaggregated_engines(decode_offset=8), trainer(num_cells=4, gpus_per_cell=8)]

        with pytest.raises(pydantic.ValidationError, match="both claim the trainer's gpu"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_refuses_a_pool_starting_part_way_into_one_of_its_own_pods(self):
        """An 8-gpu engine starting at gpu 4 would want half of each of two nodes' worth of cards."""
        specs = [engine(num_cells=1, gpus_per_engine=8, gpu_offset=4), trainer(num_cells=4, gpus_per_cell=8)]

        with pytest.raises(pydantic.ValidationError, match="not a whole number of its own"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_refuses_more_engine_cells_than_the_trainer_can_seat(self):
        """An engine rank on a gpu no trainer shares would receive nothing from a weight update."""
        specs = [engine(num_cells=8, gpus_per_engine=8), trainer(num_cells=1, gpus_per_cell=32)]

        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_rejects_a_pool_whose_cell_is_smaller_than_a_node(self):
        """The device plugin picks the cards, so a sub-node engine's base gpu id cannot be rendered."""
        specs = [engine(num_cells=1, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=4)]

        with pytest.raises(AssertionError, match="sub-node"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_refuses_a_colocated_run_whose_engines_all_sit_past_the_trainer(self):
        """Installing a pairing controller with nothing to pair would leave every engine gated forever."""
        specs = [engine(num_cells=1, gpus_per_engine=8, gpu_offset=32), trainer(num_cells=4, gpus_per_cell=8)]

        with pytest.raises(AssertionError, match="nothing to pair"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_leaves_a_run_that_does_not_colocate_without_the_section(self):
        """A disaggregated run must not gain a pairing controller with pod write rights."""
        specs = [*_disaggregated_engines(decode_offset=16), trainer(num_cells=4, gpus_per_cell=8)]

        assert "colocate" not in build_values(specs, LAYOUT).as_values()["run"]
