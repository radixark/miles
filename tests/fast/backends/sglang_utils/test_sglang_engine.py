from __future__ import annotations

import shlex
import sys

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args

pytest.importorskip("sglang")

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_plan


def _plan(*, worker_type: str = "regular", args=None, addr_overrides: dict | None = None, **kwargs):
    addr_and_ports = dict(
        host="10.0.0.1",
        port=30000,
        nccl_port=20031,
        engine_info_bootstrap_port=20033,
        dist_init_addr="10.0.0.1:20000",
    )
    addr_and_ports.update(addr_overrides or {})
    return compute_engine_launch_plan(
        args or make_engine_args(),
        node_rank=0,
        worker_type=worker_type,
        base_gpu_id=0,
        sglang_overrides={},
        num_gpus_per_engine=1,
        addr_and_ports=addr_and_ports,
        **kwargs,
    )


class TestComputeEngineLaunchPlan:
    def test_the_command_launches_sglang_with_the_allocated_addressing(self):
        """The plan renders one launch_server command carrying the addr map."""
        plan = _plan()
        tokens = shlex.split(plan.cmd)
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
        plan = _plan(addr_overrides=dict(host="[fd00::2]", port=31007, dist_init_addr="[fd00::1]:15003"))
        parsed = parse_server_args_argv(shlex.split(plan.cmd)[3:])
        assert parsed.host == "fd00::2"
        assert parsed.dist_init_addr == "[fd00::1]:15003"

    def test_a_prefill_plan_carries_the_bootstrap_port(self):
        """PD-disaggregation prefill flags survive into the command."""
        plan = _plan(worker_type="prefill", addr_overrides=dict(disaggregation_bootstrap_port=20090))
        parsed = parse_server_args_argv(shlex.split(plan.cmd)[3:])
        assert parsed.disaggregation_mode == "prefill"
        assert parsed.disaggregation_bootstrap_port == 20090

    def test_the_plan_exposes_the_api_key_for_the_health_wait(self):
        """The driver-side health wait needs the same api key the server got."""
        args = make_engine_args()
        args.sglang_api_key = "secret"
        assert _plan(args=args).api_key == "secret"

    def test_the_plan_has_no_api_key_when_the_server_has_none(self):
        """No key configured means the health wait sends none."""
        assert _plan().api_key is None
