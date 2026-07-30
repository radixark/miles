from __future__ import annotations

from argparse import Namespace
from typing import Any

import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv, server_args_to_argv
from miles.backends.sglang_utils.sglang_engine import _compute_server_args


def _args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = dict(
        hf_checkpoint="/fake/model",
        seed=42,
        offload_rollout=False,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _server_args(
    *,
    worker_type: str = "regular",
    rank: int = 0,
    dist_init_addr: str = "10.0.0.1:20000",
    args: Namespace | None = None,
    sglang_overrides: dict | None = None,
    disaggregation_bootstrap_port: int | None = None,
    num_gpus_per_engine: int = 1,
) -> ServerArgs:
    server_args_dict = _compute_server_args(
        args or _args(),
        rank,
        dist_init_addr,
        20031,
        "10.0.0.1",
        30000,
        worker_type=worker_type,
        disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        base_gpu_id=0,
        engine_info_bootstrap_port=20033,
        sglang_overrides=sglang_overrides,
        num_gpus_per_engine=num_gpus_per_engine,
    )
    return ServerArgs(**server_args_dict)


def _roundtrip(server_args: ServerArgs) -> ServerArgs:
    return parse_server_args_argv(server_args_to_argv(server_args))


class TestServerArgsToArgv:
    def test_a_regular_engine_launch_roundtrips(self):
        """The exact ServerArgs the launch computes survives the argv boundary."""
        server_args = _server_args()
        assert _roundtrip(server_args) == server_args

    def test_the_identity_flags_are_always_rendered_exactly_once(self):
        """model path and addressing must be explicit on the command, even at CLI defaults."""
        argv = server_args_to_argv(_server_args())
        for flag in ("--model-path", "--host", "--port"):
            assert argv.count(flag) == 1

    def test_a_prefill_worker_roundtrips(self):
        """PD-disaggregation prefill fields survive the argv boundary."""
        server_args = _server_args(worker_type="prefill", disaggregation_bootstrap_port=20090)
        assert server_args.disaggregation_mode == "prefill"
        assert _roundtrip(server_args) == server_args

    def test_a_decode_worker_roundtrips(self):
        """PD-disaggregation decode fields survive the argv boundary."""
        server_args = _server_args(worker_type="decode")
        assert server_args.disaggregation_mode == "decode"
        assert _roundtrip(server_args) == server_args

    def test_a_multi_node_rank_roundtrips(self):
        """nnodes, node_rank and tp_size of a multi-node engine survive the boundary."""
        server_args = _server_args(
            rank=1,
            num_gpus_per_engine=16,
            args=_args(rollout_num_gpus_per_engine=16),
        )
        assert server_args.nnodes == 2 and server_args.node_rank == 1
        assert _roundtrip(server_args) == server_args

    def test_dtype_and_parallel_sizes_roundtrip(self):
        """fp16 and dp/pp/ep sizes land in the argv and parse back."""
        server_args = _server_args(args=_args(fp16=True, sglang_dp_size=2, sglang_ep_size=2))
        assert server_args.dtype == "float16"
        assert _roundtrip(server_args) == server_args

    def test_sglang_overrides_roundtrip(self):
        """User overrides merged into the dict survive the argv boundary."""
        server_args = _server_args(sglang_overrides={"mem_fraction_static": 0.5, "log_level": "warning"})
        assert server_args.mem_fraction_static == 0.5
        assert _roundtrip(server_args) == server_args

    def test_lora_fields_roundtrip(self):
        """enable_lora, ranks and the target-modules list survive the boundary."""
        server_args = _server_args(args=_args(lora_rank=8, target_modules=["linear_qkv"]))
        assert server_args.enable_lora
        assert _roundtrip(server_args) == server_args

    def test_an_ipv6_dist_init_addr_roundtrips(self):
        """The bracketed v6 rendezvous address survives the argv boundary."""
        server_args = _server_args(dist_init_addr="[fd00::1]:20000")
        assert _roundtrip(server_args) == server_args

    def test_lora_adapter_paths_roundtrip(self):
        """The name=path lora mapping survives the argv boundary."""
        server_args = _server_args(
            args=_args(lora_rank=8, target_modules=["linear_qkv"], lora_adapter_path="/fake/adapter")
        )
        assert _roundtrip(server_args) == server_args
