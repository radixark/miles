import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import AllConfig


_BACKEND_FIELDS = frozenset(
    {
        "bf16",
        "calculate_per_token_loss",
        "ckpt_step",
        "clip_grad",
        "load",
        "mtp_detach_heads",
        "mtp_loss_scaling_factor",
        "mtp_num_layers",
        "no_load_optim",
        "no_load_rng",
        "no_save_optim",
        "rank",
        "world_size",
    }
)


class TestFsdpBackendCapture:
    @pytest.mark.parametrize("offload_train", [False, True])
    def test_native_backend_fields_survive_real_parsing_and_json_transport(self, offload_train: bool) -> None:
        """Native capture transports backend fields and conditional backups under shared offload ownership."""
        offload_flag = "--offload-train" if offload_train else "--no-offload-train"
        config = parse_fsdp_test_config("--clip-grad", "0.75", offload_flag)
        restored = AllConfig.model_validate_json(config.model_dump_json())
        backend = vars(restored.raw_fsdp)

        assert _BACKEND_FIELDS.issubset(backend)
        assert backend["clip_grad"] == 0.75
        assert backend["rank"] == 0
        assert backend["world_size"] == 1
        assert backend == vars(config.raw_fsdp)
        assert restored.offload_train is config.offload_train is offload_train
        assert "offload_train" not in backend
        for name in ("disable_grad_buffers_cpu_backup", "disable_param_buffers_cpu_backup"):
            if offload_train:
                assert backend[name] is True
            else:
                assert name not in backend
        assert not {"actor_num_nodes", "actor_num_gpus_per_node", "rollout_function_path"}.intersection(backend)
