import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestFsdpOffloadNormalization:
    @pytest.mark.parametrize("cpu_offload", [False, True])
    def test_cpu_offload_disables_train_offload_before_trainer_transport(self, cpu_offload: bool) -> None:
        """FSDP CPU offload replaces train offload before constructing an immutable trainer payload."""
        flags = ["--offload-train"]
        if cpu_offload:
            flags.append("--fsdp-cpu-offload")
        source = parse_fsdp_test_config(*flags)

        trainer = compute_trainer_config(source, source.raw_megatron.trainers[0])
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())

        assert source.offload_train is not cpu_offload
        assert source.raw_fsdp.fsdp_cpu_offload is cpu_offload
        assert restored.offload_train is not cpu_offload
        assert restored.backend.fsdp_cpu_offload is cpu_offload
        assert source.offload_train is trainer.offload_train
