from __future__ import annotations

import shlex
import sys

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args

pytest.importorskip("sglang")

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_engine import compute_api_key, compute_engine_launch_cmd


def _cmd(*, worker_type: str = "regular", args=None, addr_overrides: dict | None = None, **kwargs) -> str:
    addr_and_ports = dict(
        host="10.0.0.1",
        port=30000,
        nccl_port=20031,
        engine_info_bootstrap_port=20033,
        dist_init_addr="10.0.0.1:20000",
    )
    addr_and_ports.update(addr_overrides or {})
    return compute_engine_launch_cmd(
        args or make_engine_args(),
        node_rank=0,
        worker_type=worker_type,
        base_gpu_id=0,
        sglang_overrides={},
        num_gpus_per_engine=1,
        addr_and_ports=addr_and_ports,
        **kwargs,
    )


class TestComputeEngineLaunchCmd:
    def test_the_command_launches_sglang_with_the_allocated_addressing(self):
        """The rendered launch_server command carries the addr map."""
        tokens = shlex.split(_cmd())
        assert tokens[:3] == [sys.executable, "-m", "sglang.launch_server"]
        parsed = parse_server_args_argv(tokens[3:])
        assert parsed.host == "10.0.0.1" and parsed.port == 30000
        assert parsed.dist_init_addr == "10.0.0.1:20000"
        assert parsed.model_path == "/fake/model"

    def test_every_plan_picks_a_fresh_random_seed(self):
        """Each launch leaves the seed to sglang, so two plans never share one."""
        args = make_engine_args()
        seeds: set[int] = {parse_server_args_argv(shlex.split(_cmd(args=args))[3:]).random_seed for _ in range(5)}
        assert len(seeds) > 1
        assert args.seed not in seeds

    def test_a_bracketed_v6_host_is_stripped_for_the_server_but_kept_in_dist_addr(self):
        """sglang binds a bare v6 host while the rendezvous addr stays bracketed."""
        cmd = _cmd(addr_overrides=dict(host="[fd00::2]", port=31007, dist_init_addr="[fd00::1]:15003"))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.host == "fd00::2"
        assert parsed.dist_init_addr == "[fd00::1]:15003"

    def test_a_prefill_command_carries_the_bootstrap_port(self):
        """PD-disaggregation prefill flags survive into the command."""
        cmd = _cmd(worker_type="prefill", addr_overrides=dict(disaggregation_bootstrap_port=20090))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.disaggregation_mode == "prefill"
        assert parsed.disaggregation_bootstrap_port == 20090

    def test_the_command_carries_the_api_key_from_args(self):
        """--sglang-api-key reaches the server through the generic passthrough."""
        cmd = _cmd(args=make_engine_args(sglang_api_key="secret"))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.api_key == "secret"


class TestComputeApiKey:
    def test_the_args_key_is_used_when_no_override_exists(self):
        """The health wait needs the same key the generic passthrough gave the server."""
        args = make_engine_args(sglang_api_key="secret")
        assert compute_api_key(args, sglang_overrides={}) == "secret"

    def test_an_override_key_wins_over_the_args_key(self):
        """Overrides beat args exactly like they do in the rendered command."""
        args = make_engine_args(sglang_api_key="from-args")
        assert compute_api_key(args, sglang_overrides={"api_key": "from-override"}) == "from-override"

    def test_no_key_anywhere_means_the_health_wait_sends_none(self):
        """No key configured means the health wait sends none."""
        assert compute_api_key(make_engine_args(), sglang_overrides={}) is None
