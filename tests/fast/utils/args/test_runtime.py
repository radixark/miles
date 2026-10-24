import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestTrainerOwnership:

    def test_backend_identity_must_match_the_trainer_backend_selection(self) -> None:
        """A transported trainer cannot advertise one backend while carrying another backend's options."""
        all_config = parse_megatron_test_config()
        trainer = compute_trainer_config(all_config, trainer=all_config.raw_megatron.trainers[0])
        payload = trainer.model_dump(mode='python') | {'train_backend': 'fsdp'}
        with pytest.raises(ValueError, match='train_backend must match'):
            TrainerConfig.model_validate(payload)

    def test_duplicated_backend_fields_are_rejected_even_when_values_agree(self) -> None:
        """Equal values do not make ambiguous configuration ownership safe."""
        all_config = parse_megatron_test_config()
        trainer = compute_trainer_config(all_config, trainer=all_config.raw_megatron.trainers[0])
        payload = trainer.model_dump(mode='python')
        payload['backend'] = MegatronArgsNamespace(**vars(trainer.backend) | {'run_uuid': trainer.run_uuid})
        with pytest.raises(ValueError, match='Duplicated trainer backend fields.*run_uuid'):
            TrainerConfig.model_validate(payload)

    def test_unknown_backend_tags_are_rejected_during_transport(self) -> None:
        """A misspelled backend tag cannot fall back to another namespace type."""
        all_config = parse_megatron_test_config()
        trainer = compute_trainer_config(all_config, trainer=all_config.raw_megatron.trainers[0])
        payload = trainer.model_dump(mode='json')
        payload['backend']['backend_name'] = 'typo'
        with pytest.raises(ValueError, match='union_tag_invalid'):
            TrainerConfig.model_validate(payload)
