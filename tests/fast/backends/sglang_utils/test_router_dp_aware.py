from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import argparse

from miles.backends.sglang_utils.arguments import add_sglang_arguments, validate_args
from miles.utils.args.configs.router import RouterConfig


def _args(argv):
    parser = add_sglang_arguments(argparse.ArgumentParser())
    RouterConfig.add_arguments(parser)
    args = parser.parse_args(argv)
    args.rollout_num_gpus_per_engine = 4
    args.true_on_policy_mode = False
    args.loss_type = "policy_loss"
    args.recompute_logprobs_via_prefill = False
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
