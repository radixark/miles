from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.backends.sglang_utils.sglang_engine import _compute_server_args


def make_args(**overrides: object) -> SimpleNamespace:
    defaults = dict(
        hf_checkpoint="/fake/model",
        seed=0,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        offload_rollout=False,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_mem_fraction_static=0.7,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_adapter_path=None,
        multi_lora_n_adapters=1,
        target_modules=["linear_qkv"],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def compute(args: SimpleNamespace, **overrides: object) -> dict:
    kwargs = dict(
        node_rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=5000,
        host="127.0.0.1",
        port=30000,
        worker_type="regular",
        disaggregation_bootstrap_port=None,
        base_gpu_id=0,
        engine_info_bootstrap_port=None,
        sglang_overrides=None,
        num_gpus_per_engine=None,
        gated_launch_port=30001,
    )
    kwargs.update(overrides)
    return _compute_server_args(args, **kwargs)


class TestRandomSeed:
    def test_engine_args_do_not_force_a_random_seed(self):
        """Every engine must be free to pick its own seed, so the launch args must not carry one."""
        args = make_args(seed=1234)

        server_args = compute(args)

        assert "random_seed" not in server_args


class TestSglangOverridePrecedence:
    """An override must win over every args-derived default, including the conditional ones."""

    def test_override_wins_over_conditional_args_defaults(self):
        args = make_args(fp16=True, use_rollout_routing_replay=True, use_rollout_indexer_replay=True)

        server_args = compute(args, sglang_overrides={"dtype": "bfloat16"})

        assert server_args["dtype"] == "bfloat16"

    def test_override_wins_over_lora_defaults(self):
        args = make_args(lora_rank=8)

        server_args = compute(args, sglang_overrides={"enable_lora": False})

        assert server_args["enable_lora"] is False

    @pytest.mark.parametrize("value", [0.5, 0.95])
    def test_override_wins_over_base_sglang_args(self, value):
        args = make_args(sglang_mem_fraction_static=0.7)

        server_args = compute(args, sglang_overrides={"mem_fraction_static": value})

        assert server_args["mem_fraction_static"] == value

    def test_no_overrides_keeps_args_derived_values(self):
        args = make_args(fp16=True, lora_rank=8)

        server_args = compute(args)

        assert server_args["dtype"] == "float16"
        assert server_args["enable_lora"] is True
        assert server_args["mem_fraction_static"] == 0.7
