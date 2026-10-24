import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config


class TestFsdpOffloadNormalization:
    @pytest.mark.parametrize("cpu_offload", [False, True])
    def test_cpu_offload_disables_train_offload_before_trainer_transport(self, cpu_offload: bool) -> None:
        """FSDP CPU offload replaces train offload during parser normalization."""
        flags = ["--offload-train"]
        if cpu_offload:
            flags.append("--fsdp-cpu-offload")
        source = parse_fsdp_test_config(*flags)

        assert source.offload_train is not cpu_offload
        assert source.raw_megatron.base_args["fsdp_cpu_offload"] is cpu_offload
