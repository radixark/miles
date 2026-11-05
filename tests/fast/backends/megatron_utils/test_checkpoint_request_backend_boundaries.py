import json
from pathlib import Path

from tests.fast.fixtures.args_fixtures import _COMMON_TEST_ARGV, _parse_test_config, parse_fsdp_test_config

from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig
from miles.backends.megatron_utils.megatron_config import MegatronTrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_ckpt_load_config, compute_trainer_config


class TestCheckpointRequestBackendBoundaries:
    def test_fsdp_has_no_megatron_loading_request_and_keeps_its_configuration(self) -> None:
        """Non-Megatron checkpoint handling cannot inherit Megatron runtime loading inputs."""
        args = parse_fsdp_test_config()
        trainer = MegatronTrainerConfig(trainer_id="actor", model_id=None, role="actor", actor_index=0, overrides={})
        original = args.model_dump(mode="python")

        assert compute_trainer_ckpt_load_config(args, trainer) is None
        assert args.model_dump(mode="python") == original

    def test_torchtitan_resume_keeps_its_load_and_start_rollout(self, tmp_path: Path) -> None:
        """A --load holding a torchtitan save is not rewritten to --ref-load: the Megatron
        resolver (no tracker file means a fresh run) does not apply to torchtitan."""
        load = tmp_path / "run"
        (load / "torchtitan" / "actor" / "checkpoint" / "step-4").mkdir(parents=True)
        hf = tmp_path / "hf"
        hf.mkdir()
        hf_config = {
            "model_type": "qwen2",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 128,
        }
        (hf / "config.json").write_text(json.dumps(hf_config))
        args = _parse_test_config(
            [
                "--train-backend",
                "torchtitan",
                *_COMMON_TEST_ARGV,
                "--load",
                str(load),
                "--ref-load",
                str(hf),
                "--hf-checkpoint",
                str(hf),
                "--seq-length",
                "64",
                "--rollout-max-response-len",
                "8",
            ]
        )
        trainer = MegatronTrainerConfig(trainer_id="actor", model_id=None, role="actor", actor_index=0, overrides={})

        assert compute_trainer_ckpt_load_config(args, trainer) is None
        trainer_config = compute_trainer_config(args, trainer)
        assert trainer_config.backend.load == str(load)
        assert trainer_config.start_rollout_id is None
        assert args.load == str(load)

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
