"""True-on-policy deterministic variant of run_qwen3_30b_a3b.

Extends the base script with true-on-policy contract, EP parity defaults,
and deterministic launch plan injection. Use this script when you need
exact-zero logprob alignment between SGLang rollout and Megatron training.

This module owns every true-on-policy concern (topology-suffixed torch_dist
checkpoints, vocab padding overrides, fused-grad workaround, debug-one-sample
plumbing). The base ``scripts/run_qwen3_30b_a3b`` script stays
true-on-policy-agnostic; we cooperate with it through ``prepare``'s
``convert_checkpoint_kwargs`` hook and by appending overrides at the end of
``args.extra_args`` so argparse last-wins picks them up.
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer
from scripts.run_qwen3_30b_a3b import ScriptArgs as BaseScriptArgs
from scripts.run_qwen3_30b_a3b import prepare as base_prepare

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
        rollout_engine_size_was_default = self.rollout_num_gpus_per_engine is None
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
            and rollout_engine_size_was_default
        ):
            self.rollout_num_gpus_per_engine = (
                self.sglang_expert_parallel_size * self.expert_tensor_parallel_size
            )
        apply_true_on_policy_script_defaults(self)


def _uses_topology_aware_torch_dist(args: ScriptArgs) -> bool:
    return bool(args.true_on_policy and args.expert_model_parallel_size > 1)


def _megatron_torch_dist_path(args: ScriptArgs) -> str:
    if _uses_topology_aware_torch_dist(args):
        topology = (
            f"tp{args.tensor_model_parallel_size}"
            f"_pp{args.pipeline_model_parallel_size}"
            f"_ep{args.expert_model_parallel_size}"
            f"_etp{args.expert_tensor_parallel_size}"
        )
        return f"{args.model_dir}/{args.model_name}_torch_dist_{topology}"
    return f"{args.model_dir}/{args.model_name}_torch_dist"


def _megatron_torch_dist_conversion_args(args: ScriptArgs) -> str:
    if not _uses_topology_aware_torch_dist(args):
        return ""
    return (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} "
        f"--expert-tensor-parallel-size {args.expert_tensor_parallel_size} "
        f"{_true_on_policy_vocab_padding_args(args)}"
    )


def _true_on_policy_vocab_padding_args(args: ScriptArgs) -> str:
    if not (
        args.true_on_policy
        and args.expert_model_parallel_size > 1
        and args.tensor_model_parallel_size > 1
    ):
        return ""
    # Qwen3-30B-A3B's real vocab is already divisible by supported TP sizes.
    # Avoid Megatron's default extra padding so HF -> torch-dist conversion and
    # true-on-policy scoring use the same shard width.
    return "--make-vocab-size-divisible-by 1 "


def _true_on_policy_sequence_parallel_backward_args(args: ScriptArgs) -> str:
    if not (
        args.true_on_policy
        and args.expert_model_parallel_size > 1
        and args.tensor_model_parallel_size > 1
        and args.use_sequence_parallel
    ):
        return ""
    # TP+EP sequence-parallel true-on-policy currently produces nonfinite local
    # wgrad buckets with Megatron's fused gradient accumulation path. Keep the
    # unfused path for correctness until the fused kernel path is audited.
    return "--no-gradient-accumulation-fusion "


def _topology_aware_extra_args(args: ScriptArgs) -> str:
    """Args injected at the end of train_args to override the base script.

    Relies on argparse last-wins semantics for ``--load`` / ``--ref-load`` so
    the topology-suffixed torch_dist directory wins over the base's default.
    """
    if not _uses_topology_aware_torch_dist(args):
        return ""
    load_path = _megatron_torch_dist_path(args)
    parts = [
        f"--load {load_path}",
        f"--ref-load {load_path}",
    ]
    vocab = _true_on_policy_vocab_padding_args(args).strip()
    if vocab:
        parts.append(vocab)
    grad = _true_on_policy_sequence_parallel_backward_args(args).strip()
    if grad:
        parts.append(grad)
    return " ".join(parts) + " "


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


def prepare(args: ScriptArgs):
    convert_kwargs: dict = {}
    if _uses_topology_aware_torch_dist(args):
        convert_kwargs = {
            "path_dst": _megatron_torch_dist_path(args),
            "extra_args": _megatron_torch_dist_conversion_args(args),
        }
    base_prepare(args, convert_checkpoint_kwargs=convert_kwargs)


def execute(args: ScriptArgs):
    from scripts.run_qwen3_30b_a3b import execute as base_execute

    plan = build_true_on_policy_launch_plan(args)
    os.environ.update(plan.env_vars)
    plan_env_vars = " ".join(f"{key}={value}" for key, value in plan.env_vars.items())
    args.extra_env_vars = " ".join(part for part in (plan_env_vars, args.extra_env_vars) if part)
    debug_args = _debug_one_sample_args(args) if args.mode == "debug_one_sample" else ""
    if args.mode == "debug_one_sample":
        args.enable_eval = False
    topology_extra = _topology_aware_extra_args(args)
    base_mode = args.mode
    args.mode = "debug_minimal" if args.mode == "debug_one_sample" else args.mode
    args.extra_args = f"{plan.train_args} {debug_args} {topology_extra} {args.extra_args}"
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
