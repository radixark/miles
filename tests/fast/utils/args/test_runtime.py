import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import (
    AllConfig,
    InferenceControllerConfig,
    OrchestratorConfig,
    RolloutConfig,
    TrainerConfig,
)
from miles.utils.args.trainer_utils import compute_trainer_config


_DEPLOYMENT_FIELDS = frozenset(
    {
        "actor_num_nodes",
        "actor_num_gpus_per_node",
        "critic_num_nodes",
        "critic_num_gpus_per_node",
        "rollout_num_gpus",
        "eval_num_gpus",
        "eval_num_gpus_per_engine",
    }
)


class TestTrainerOwnership:
    def test_backend_identity_must_match_the_trainer_backend_selection(self) -> None:
        """A transported trainer cannot advertise one backend while carrying another backend's options."""
        all_config = parse_megatron_test_config()
        trainer = compute_trainer_config(all_config, all_config.raw_megatron.trainers[0])
        payload = trainer.model_dump(mode="python") | {"train_backend": "fsdp"}

        with pytest.raises(ValueError, match="train_backend must match"):
            TrainerConfig.model_validate(payload)

    def test_duplicated_backend_fields_are_rejected_even_when_values_agree(self) -> None:
        """Equal values do not make ambiguous configuration ownership safe."""
        all_config = parse_megatron_test_config()
        trainer = compute_trainer_config(all_config, all_config.raw_megatron.trainers[0])
        payload = trainer.model_dump(mode="python")
        payload["backend"] = MegatronArgsNamespace(**(vars(trainer.backend) | {"run_uuid": trainer.run_uuid}))

        with pytest.raises(ValueError, match="Duplicated trainer backend fields.*run_uuid"):
            TrainerConfig.model_validate(payload)

    def test_unknown_backend_tags_are_rejected_during_transport(self) -> None:
        """A misspelled backend tag cannot fall back to another namespace type."""
        all_config = parse_megatron_test_config()
        trainer = compute_trainer_config(all_config, all_config.raw_megatron.trainers[0])
        payload = trainer.model_dump(mode="json")
        payload["backend"]["backend_name"] = "typo"

        with pytest.raises(ValueError, match="union_tag_invalid"):
            TrainerConfig.model_validate(payload)

    def test_same_name_delegates_follow_backend_values_without_duplicating_storage(self) -> None:
        """Shared consumers read their trainer's backend values rather than stale flat copies."""
        backend = MegatronArgsNamespace(
            global_batch_size=16,
            micro_batch_size=3,
            wandb_project="project",
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=4,
            context_parallel_size=8,
        )
        trainer = TrainerConfig.model_construct(backend=backend)

        assert trainer.global_batch_size == 16
        assert trainer.micro_batch_size == 3
        assert trainer.wandb_project == "project"
        assert trainer.tensor_model_parallel_size == 2
        assert trainer.pipeline_model_parallel_size == 4
        assert trainer.context_parallel_size == 8
        assert not set(vars(backend)).intersection(TrainerConfig.model_fields)


class TestComponentTransport:
    @pytest.mark.parametrize("component", [OrchestratorConfig, RolloutConfig, InferenceControllerConfig])
    def test_sliced_components_roundtrip_without_deployment_scaling_fields(self, component: type) -> None:
        """Component payloads survive transport while deployment-only fields stay with the launcher."""
        all_config = parse_megatron_test_config()
        leaf = component.slice_from(all_config)
        restored = component.model_validate_json(leaf.model_dump_json())

        assert restored.model_dump(mode="json") == leaf.model_dump(mode="json")
        assert not _DEPLOYMENT_FIELDS.intersection(restored.model_dump())
        assert _DEPLOYMENT_FIELDS.issubset(all_config.model_dump())
        assert "sglang_scaling" not in restored.model_dump()
        assert all_config.actor_num_nodes == 1

    def test_trainer_transport_does_not_hide_deployment_fields_in_its_backend(self) -> None:
        """The trainer payload cannot retain scaling at either its top level or its backend namespace."""
        all_config = parse_megatron_test_config()
        trainer = compute_trainer_config(all_config, all_config.raw_megatron.trainers[0])
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())
        payload = restored.model_dump(mode="json")

        assert not _DEPLOYMENT_FIELDS.intersection(payload)
        assert not _DEPLOYMENT_FIELDS.intersection(payload["backend"])
        assert _DEPLOYMENT_FIELDS.issubset(all_config.model_dump())

    def test_unknown_all_config_fields_are_rejected(self) -> None:
        """The complete run configuration rejects stale or misspelled serialized fields."""
        config = parse_megatron_test_config()
        with pytest.raises(ValueError, match="extra_forbidden"):
            AllConfig.model_validate(config.model_dump(mode="python") | {"unknown_option": True})
