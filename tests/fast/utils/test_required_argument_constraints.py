from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.utils.test_arguments import DYNAMIC_BATCH_ARGS, _parse_megatron_args

from miles.utils.arguments import miles_validate_args
from miles.utils.lora.arguments import validate_multi_lora_args


class TestRequiredArgumentConstraints:
    @pytest.mark.parametrize("explicit_directory", [False, True])
    def test_train_healing_supplies_only_a_missing_local_checkpoint_directory(
        self, tmp_path: Path, explicit_directory: bool
    ) -> None:
        """Train healing preserves a requested checkpoint location and derives its atomic storage settings."""
        directory = tmp_path / "local-checkpoints"
        extra = ["--non-persistent-local-ckpt-dir", str(directory)] if explicit_directory else []
        args = _parse_megatron_args(["--use-fault-tolerance", "--ft-components", "train", *DYNAMIC_BATCH_ARGS, *extra])

        miles_validate_args(args)

        assert args.non_persistent_local_ckpt_dir == (
            str(directory) if explicit_directory else "/tmp/miles_local_ckpt"
        )
        assert args.non_persistent_ckpt_type == "local"
        assert args.non_persistent_local_ckpt_algo == "atomic"

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_flops_balancing_requires_the_dynamic_scheduler(self, dynamic: bool) -> None:
        """The parser refuses FLOPs balancing without the scheduler that consumes its workloads."""
        args = _parse_megatron_args(["--balance-by-flops", *(DYNAMIC_BATCH_ARGS if dynamic else [])])

        if not dynamic:
            with pytest.raises(AssertionError, match="--balance-by-flops requires --use-dynamic-batch-size"):
                miles_validate_args(args)
            return
        miles_validate_args(args)

        assert args.balance_by_flops and args.use_dynamic_batch_size

    @pytest.mark.parametrize("transfer", ["p2p", "disk-delta"])
    def test_partial_weight_transfers_reject_disaggregated_prefill_before_storage_setup(self, transfer: str) -> None:
        """Unsupported PD transfers fail on topology before disk paths or checkpoint assets are accessed."""
        args = _parse_megatron_args(["--update-weight-transfer-mode", transfer, "--prefill-num-servers", "1"])

        with pytest.raises(AssertionError, match="has not been tested when PD is enabled"):
            miles_validate_args(args)

    @pytest.mark.parametrize("dispatcher", ["allgather", "alltoall"])
    def test_variable_length_training_retains_a_compatible_backend_dispatcher(self, dispatcher: str) -> None:
        """Real parsing normalizes incompatible dispatchers before the backend is transported."""
        config = parse_megatron_test_config("--moe-token-dispatcher-type", dispatcher)

        assert config.raw_megatron.base_args["moe_token_dispatcher_type"] == "alltoall"
        assert config.raw_megatron.base_args["variable_seq_lengths"] is True

    def test_a_server_group_config_rejects_a_second_prefill_topology_before_reading_the_file(
        self, tmp_path: Path
    ) -> None:
        """Conflicting topology owners are rejected before a missing server configuration can be opened."""
        args = _parse_megatron_args(["--sglang-config", str(tmp_path / "absent.yaml"), "--prefill-num-servers", "1"])

        with pytest.raises(AssertionError, match="sglang_config and prefill_num_servers are mutually exclusive"):
            miles_validate_args(args)

    @pytest.mark.parametrize("field,message", [("indep_dp", "independent-DP"), ("enable_witness", "witness module")])
    def test_multi_adapter_training_refuses_state_machines_that_assume_a_shared_optimizer(
        self, field: str, message: str
    ) -> None:
        """Multi-adapter validation rejects incompatible optimizer lifecycles before model construction."""
        args = _parse_megatron_args(["--multi-lora-n-adapters", "2", "--lora-rank", "8"])
        args.ft_components = []
        setattr(args, field, True)

        with pytest.raises(AssertionError, match=message):
            validate_multi_lora_args(args)
