import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config, parse_megatron_test_config

from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestTrainerBackendTraits:
    @pytest.mark.parametrize("backend", ["megatron", "fsdp"])
    def test_shared_training_traits_are_stored_once_in_the_transported_backend(self, backend: str) -> None:
        """Each backend receives shared training settings without duplicating their storage in the trainer."""
        parse = {"megatron": parse_megatron_test_config, "fsdp": parse_fsdp_test_config}[backend]
        config = parse(
            "--seed",
            "37",
            "--global-batch-size",
            "2",
            "--wandb-project",
            "backend-traits",
            "--distributed-timeout-minutes",
            "12",
        )
        before = config.model_dump(mode="json")
        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())
        values = vars(restored.backend)
        expected = {
            "seed": 37,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "wandb_project": "backend-traits",
            "distributed_timeout_minutes": 12,
        }

        assert expected.items() <= values.items()
        assert not expected.keys() & TrainerConfig.model_fields.keys()
        assert config.model_dump(mode="json") == before
