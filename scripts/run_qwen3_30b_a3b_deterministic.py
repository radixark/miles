"""True-on-policy deterministic variant of run_qwen3_30b_a3b.

Extends the base script with true-on-policy contract, EP parity defaults,
and deterministic launch plan injection. Use this script when you need
exact-zero logprob alignment between SGLang rollout and Megatron training.
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer
from scripts.run_qwen3_30b_a3b import ScriptArgs as BaseScriptArgs
from scripts.run_qwen3_30b_a3b import prepare

import miles.utils.external_utils.command_utils as U
from miles.true_on_policy import apply_true_on_policy_script_defaults, build_true_on_policy_launch_plan


@dataclass
class ScriptArgs(BaseScriptArgs):
    mode: Literal["normal", "debug_minimal", "debug_one_sample"] = "normal"
    train_backend: Literal["megatron"] = "megatron"
    true_on_policy: bool = True
    true_on_policy_contract: str | None = None
    true_on_policy_default_rollout_ep: bool = True

    def __post_init__(self):
        super().__post_init__()
        if (
            self.sglang_expert_parallel_size == 1
            and self.true_on_policy
            and self.true_on_policy_default_rollout_ep
        ):
            self.sglang_expert_parallel_size = self.expert_model_parallel_size
        if (
            self.true_on_policy
            and self.sglang_expert_parallel_size > 1
            and self.rollout_num_gpus_per_engine < self.sglang_expert_parallel_size
        ):
            self.rollout_num_gpus_per_engine = self.sglang_expert_parallel_size
        apply_true_on_policy_script_defaults(self)


def _debug_one_sample_args(args: ScriptArgs) -> str:
    train_data_parallel_size = (
        args.num_nodes
        * args.num_gpus_per_node
        // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size)
    )
    debug_batch_size = max(1, train_data_parallel_size)
    return (
        "--num-rollout 1 "
        f"--rollout-batch-size {debug_batch_size} "
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 2 "
        f"--global-batch-size {debug_batch_size} "
        "--ci-test --ci-disable-kl-checker "
        "--sglang-disable-cuda-graph "
    )


def execute(args: ScriptArgs):
    from scripts.run_qwen3_30b_a3b import execute as base_execute

    plan = build_true_on_policy_launch_plan(args)
    os.environ.update(plan.env_vars)
    plan_env_vars = " ".join(f"{key}={value}" for key, value in plan.env_vars.items())
    args.extra_env_vars = " ".join(part for part in (plan_env_vars, args.extra_env_vars) if part)
    debug_args = _debug_one_sample_args(args) if args.mode == "debug_one_sample" else ""
    if args.mode == "debug_one_sample":
        args.enable_eval = False
    base_mode = args.mode
    args.mode = "debug_minimal" if args.mode == "debug_one_sample" else args.mode
    args.extra_args = f"{plan.train_args} {debug_args} {args.extra_args}"
    try:
        base_execute(args)
    finally:
        args.mode = base_mode


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
