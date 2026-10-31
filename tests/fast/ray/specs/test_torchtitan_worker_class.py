import pytest
import torch
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.ray.specs.train import TrainerSpec
from miles.utils.args.trainer_utils import compute_trainer_config


class TestTorchTitanWorkerClass:
    def test_native_torchtitan_configuration_selects_its_own_trainer_actor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TorchTitan trainer specifications retain their native actor after configuration slicing."""
        monkeypatch.setattr(torch, "__version__", "2.13.0")
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")
        config = parse_fsdp_test_config(
            "--train-backend", "torchtitan", "--seq-length", "32", "--rollout-max-response-len", "16"
        )
        worker_config = compute_trainer_config(config, config.raw_megatron.trainers[0])

        spec = TrainerSpec.create(worker_config)

        assert spec.worker_class == "miles.backends.torchtitan_utils.actor.TorchtitanTrainRayActor"
        assert spec.args.backend.backend_name == "torchtitan"
