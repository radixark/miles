from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig
from miles.backends.megatron_utils.megatron_config import MegatronTrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_ckpt_load_config


class TestCheckpointRequestBackendBoundaries:
    def test_fsdp_has_no_megatron_loading_request_and_keeps_its_configuration(self) -> None:
        """Non-Megatron checkpoint handling cannot inherit Megatron runtime loading inputs."""
        args = parse_fsdp_test_config()
        trainer = MegatronTrainerConfig(trainer_id="actor", model_id=None, role="actor", actor_index=0, overrides={})
        original = args.model_dump(mode="python")

        assert compute_trainer_ckpt_load_config(args, trainer) is None
        assert args.model_dump(mode="python") == original

    def test_peer_recovery_restores_training_state_without_a_disk_source(self) -> None:
        """Peer recovery restores optimizer and RNG state while relying on its in-memory source."""
        assert MegatronCkptLoadConfig.for_peer_recovery().model_dump() == {
            "load": None,
            "resume_from_ckpt": True,
            "no_load_optim": False,
            "no_load_rng": False,
            "finetune": False,
            "ckpt_step": None,
        }
