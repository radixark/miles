"""Tests for process_identity module."""

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.utils.audit_utils.process_identity import (
    ProcessIdentity,
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)


class TestProcessIdentityToName:
    def test_main(self) -> None:
        assert SimpleProcessIdentity(component="main").to_name() == "main"

    def test_rollout_executor(self) -> None:
        assert SimpleProcessIdentity(component="rollout_executor").to_name() == "rollout_executor"

    def test_actor(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=1, rank_within_cell=3)
        assert source.to_name() == "actor_cell1_rank3"

    def test_critic(self) -> None:
        source = TrainProcessIdentity(component="critic", cell_index=0, rank_within_cell=2)
        assert source.to_name() == "critic_cell0_rank2"

    def test_trainer_controller(self) -> None:
        assert TrainerControllerProcessIdentity(role="actor").to_name() == "trainer_controller_actor"

    def test_inference_controller(self) -> None:
        assert SimpleProcessIdentity(component="inference_controller").to_name() == "inference_controller"

    def test_multi_lora_controller(self) -> None:
        assert SimpleProcessIdentity(component="multi_lora_controller").to_name() == "multi_lora_controller"

    def test_worker_manager(self) -> None:
        assert SimpleProcessIdentity(component="worker_manager").to_name() == "worker_manager"

    def test_an_unknown_component_is_rejected(self) -> None:
        """A simple identity only names the components that exist."""
        with pytest.raises(ValidationError):
            SimpleProcessIdentity(component="nope")


class TestControllerIdentityRoundtrip:
    def test_trainer_controller_keeps_its_role(self) -> None:
        """Two trainer controllers share a component, so only the role tells their events apart."""
        source = TrainerControllerProcessIdentity(role="critic")
        assert TrainerControllerProcessIdentity.model_validate_json(source.model_dump_json()) == source


class TestTrainProcessIdentityValidation:
    def test_negative_cell_index_rejected(self) -> None:
        """A negative cell_index fails validation."""
        with pytest.raises(ValidationError):
            TrainProcessIdentity(component="actor", cell_index=-1, rank_within_cell=0)

    def test_negative_rank_within_cell_rejected(self) -> None:
        """A negative rank_within_cell fails validation."""
        with pytest.raises(ValidationError):
            TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=-1)


class TestProcessIdentityUnion:
    def test_process_identity_union_deserializes_rollout_executor(self) -> None:
        """The discriminated union maps the wire component "rollout_executor" to RolloutExecutorProcessIdentity."""
        parsed = TypeAdapter(ProcessIdentity).validate_python({"component": "rollout_executor"})

        assert isinstance(parsed, RolloutExecutorProcessIdentity)
        assert parsed.to_name() == "rollout_executor"

    def test_rollout_executor_survives_a_union_json_roundtrip(self) -> None:
        """A serialized RolloutExecutorProcessIdentity is parsed back to the same type through the union."""
        adapter = TypeAdapter(ProcessIdentity)

        parsed = adapter.validate_json(RolloutExecutorProcessIdentity().model_dump_json())

        assert parsed == RolloutExecutorProcessIdentity()

    def test_unknown_component_is_rejected_by_the_union(self) -> None:
        """An unknown component discriminator fails validation instead of falling back to a member."""
        with pytest.raises(ValidationError):
            TypeAdapter(ProcessIdentity).validate_python({"component": "rollout_manager"})


class TestTrainProcessIdentityRoundtrip:
    def test_serialize_deserialize(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=0)
        parsed = TrainProcessIdentity.model_validate_json(source.model_dump_json())
        assert parsed.cell_index == 2
        assert parsed.rank_within_cell == 0
        assert parsed.component == "actor"
