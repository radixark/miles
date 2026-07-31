from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import argparse

from miles.backends.sglang_utils.arguments import add_sglang_arguments, validate_args
from miles.utils.http_utils import router_worker_base_urls


def _args(argv):
    parser = add_sglang_arguments(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    args.rollout_num_gpus_per_engine = 4
    args.true_on_policy_mode = False
    args.use_session_server = False
    # Registered by RouterArgs.add_cli_args, not by add_sglang_arguments.
    args.router_assignment_mode = "random"
    args.router_dp_aware = False
    validate_args(args)
    return args


def test_dp_attention_turns_on_router_dp_aware():
    assert _args(["--sglang-enable-dp-attention", "--sglang-dp-size", "4"]).router_dp_aware is True


def test_router_dp_aware_untouched_without_dp_attention():
    assert _args([]).router_dp_aware is False


def test_dp_rank_suffix_stripped_and_engines_deduplicated():
    # dp-aware routing reports one entry per DP rank; they share one addressable server.
    assert router_worker_base_urls(["http://h:8080@0", "http://h:8080@1"]) == ["http://h:8080"]


def test_plain_worker_urls_pass_through_in_order():
    assert router_worker_base_urls(["http://h:8080", "http://h:8081"]) == ["http://h:8080", "http://h:8081"]


def test_ipv6_dp_rank_suffix_stripped():
    assert router_worker_base_urls(["http://[::1]:8080@0"]) == ["http://[::1]:8080"]


def test_non_numeric_suffix_is_not_a_dp_rank():
    # userinfo, not a rank: stripping it would change which host is addressed.
    assert router_worker_base_urls(["http://user:pass@h:8080"]) == ["http://user:pass@h:8080"]
