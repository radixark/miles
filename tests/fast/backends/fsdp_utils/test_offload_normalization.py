import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestFsdpOffloadNormalization:
    @pytest.mark.parametrize(
        ("offload_flags", "cpu_offload"),
        [((), False), (("--optimizer-cpu-offload",), True), (("--fsdp-cpu-offload",), True)],
        ids=["off", "optimizer-cpu-offload", "deprecated-fsdp-cpu-offload"],
    )
    def test_cpu_offload_disables_train_offload_before_trainer_transport(
        self, offload_flags: tuple[str, ...], cpu_offload: bool
    ) -> None:
        """FSDP CPU offload, under either spelling, replaces train offload before the immutable trainer payload."""
        source = parse_fsdp_test_config("--offload-train", *offload_flags)

        trainer = compute_trainer_config(source, source.raw_megatron.trainers[0])
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())

        assert source.offload_train is not cpu_offload
        assert source.raw_fsdp.optimizer_cpu_offload is cpu_offload
        assert restored.offload_train is not cpu_offload
        assert restored.backend.optimizer_cpu_offload is cpu_offload
        assert source.offload_train is trainer.offload_train
