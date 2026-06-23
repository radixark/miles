"""Tests for process_identity module."""

from miles.utils.process_identity import MainProcessIdentity, RolloutManagerProcessIdentity, TrainProcessIdentity


class TestProcessIdentityToName:
    def test_main(self) -> None:
        assert MainProcessIdentity().to_name() == "main"

    def test_rollout_manager(self) -> None:
        assert RolloutManagerProcessIdentity().to_name() == "rollout_manager"

    def test_actor(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=1, rank_within_cell=3)
        assert source.to_name() == "actor_cell1_rank3"

    def test_critic(self) -> None:
        source = TrainProcessIdentity(component="critic", cell_index=0, rank_within_cell=2)
        assert source.to_name() == "critic_cell0_rank2"


class TestTrainProcessIdentityRoundtrip:
    def test_serialize_deserialize(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=0)
        parsed = TrainProcessIdentity.model_validate_json(source.model_dump_json())
        assert parsed.cell_index == 2
        assert parsed.rank_within_cell == 0
        assert parsed.component == "actor"
