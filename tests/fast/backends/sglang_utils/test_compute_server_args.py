from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.utils.lora import LORA_ADAPTER_NAME


def make_args(**overrides) -> SimpleNamespace:
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


def compute(args, **kwargs) -> dict:
    server_args, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=5000,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
        **kwargs,
    )
    return server_args


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


class TestLoraAdapterPreload:
    """Startup preload is optional: the adapter also arrives through the first weight sync."""

    def test_preloads_a_directory_sglang_can_read(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        args = make_args(lora_rank=8, lora_adapter_path=str(tmp_path))

        server_args = compute(args)

        assert server_args["lora_paths"] == {LORA_ADAPTER_NAME: str(tmp_path)}

    def test_skips_a_native_only_checkpoint(self, tmp_path):
        # Bridge declines the HF export for some MoE adapter layouts, and SGLang
        # aborts at startup on the missing adapter_config.json.
        (tmp_path / "adapter_megatron_tp0_pp0_ep0.pt").write_bytes(b"")
        args = make_args(lora_rank=8, lora_adapter_path=str(tmp_path))

        server_args = compute(args)

        assert "lora_paths" not in server_args
