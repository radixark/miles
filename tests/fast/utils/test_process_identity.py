"""Tests for process_identity module."""

import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.process_identity import (
    InferenceControllerProcessIdentity,
    MainProcessIdentity,
    MultiLoRAControllerProcessIdentity,
    RolloutExecutorProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)


class TestProcessIdentityToName:
    def test_main(self) -> None:
        assert MainProcessIdentity().to_name() == "main"

    def test_rollout_executor(self) -> None:
        assert RolloutExecutorProcessIdentity().to_name() == "rollout_executor"

    def test_actor(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_id="trainer-actor-1", rank_within_cell=3)
        assert source.to_name() == "actor_trainer-actor-1_rank3"

    def test_critic(self) -> None:
        source = TrainProcessIdentity(component="critic", cell_id="trainer-critic-0", rank_within_cell=2)
        assert source.to_name() == "critic_trainer-critic-0_rank2"

    def test_trainer_controller(self) -> None:
        assert TrainerControllerProcessIdentity(role="actor").to_name() == "trainer_controller_actor"

    def test_inference_controller(self) -> None:
        assert InferenceControllerProcessIdentity().to_name() == "inference_controller"

    def test_multi_lora_controller(self) -> None:
        assert MultiLoRAControllerProcessIdentity().to_name() == "multi_lora_controller"


class TestControllerIdentityRoundtrip:
    def test_trainer_controller_keeps_its_role(self) -> None:
        """Two trainer controllers share a component, so only the role tells their events apart."""
        source = TrainerControllerProcessIdentity(role="critic")
        assert TrainerControllerProcessIdentity.model_validate_json(source.model_dump_json()) == source


class TestTrainProcessIdentityValidation:
    def test_empty_cell_id_still_parses(self) -> None:
        """A cell id is opaque, so nothing about its shape can be validated here."""
        assert TrainProcessIdentity(component="actor", cell_id="", rank_within_cell=0).cell_id == ""

    def test_negative_rank_within_cell_rejected(self) -> None:
        """A negative rank_within_cell fails validation."""
        with pytest.raises(ValidationError):
            TrainProcessIdentity(component="actor", cell_id="trainer-actor-0", rank_within_cell=-1)


class TestTrainProcessIdentityRoundtrip:
    def test_serialize_deserialize(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_id="trainer-actor-2", rank_within_cell=0)
        parsed = TrainProcessIdentity.model_validate_json(source.model_dump_json())
        assert parsed.cell_id == "trainer-actor-2"
        assert parsed.rank_within_cell == 0
        assert parsed.component == "actor"
