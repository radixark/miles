import pytest

from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode


def _mode(
    *,
    train_gpus_per_node: int,
    rollout_num_engines: int = 0,
    rollout_gpus_per_engine: int = 0,
    colocate: bool = False,
) -> FTTestMode:
    return FTTestMode(
        model_name="model",
        model_hf_repo="org/model",
        megatron_model_type="model-type",
        num_cells=2,
        parallel_args="",
        train_gpus_per_node=train_gpus_per_node,
        rollout_num_engines=rollout_num_engines,
        rollout_gpus_per_engine=rollout_gpus_per_engine,
        colocate=colocate,
    )


class TestTotalNodeGpus:
    def test_colocated_mode_counts_shared_gpus_once(self) -> None:
        """The registered colocated mode reserves only the trainer's gpus, not trainer plus rollout."""
        mode = MODES["kill_rollout__dp2_cp2__colocate"]

        assert mode.colocate
        assert mode.total_rollout_gpus == 4
        assert mode.total_node_gpus == 4

    def test_disaggregated_mode_adds_rollout_gpus_to_train_gpus(self) -> None:
        """Without colocation the rollout engines need their own gpus on top of the trainer's."""
        mode = MODES["kill_train__dp2_cp2__moe_5layer"]

        assert not mode.colocate
        assert mode.total_node_gpus == 8


class TestColocationValidation:
    def test_colocated_mode_rejects_rollout_gpu_oversubscription(self) -> None:
        """A colocated mode whose engines need more gpus than the trainer owns is rejected at construction."""
        with pytest.raises(AssertionError, match="oversubscribes its node"):
            _mode(
                train_gpus_per_node=2,
                rollout_num_engines=4,
                rollout_gpus_per_engine=1,
                colocate=True,
            )

    def test_colocated_mode_accepts_rollout_gpus_that_exactly_fill_the_node(self) -> None:
        """Rollout demand equal to the trainer's gpu count still fits, so construction succeeds."""
        mode = _mode(
            train_gpus_per_node=4,
            rollout_num_engines=4,
            rollout_gpus_per_engine=1,
            colocate=True,
        )

        assert mode.total_node_gpus == 4

    def test_a_disaggregated_mode_may_ask_for_more_rollout_gpus_than_train_gpus(self) -> None:
        """The oversubscription check applies only to colocated modes."""
        mode = _mode(
            train_gpus_per_node=2,
            rollout_num_engines=4,
            rollout_gpus_per_engine=1,
        )

        assert mode.total_node_gpus == 6
