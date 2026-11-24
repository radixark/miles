"""FSDP training-curve validation: Qwen3-0.6B (model_type=qwen3, smallest of the series), single node.

dapo-math-17k @ 4k response len, AIME-2024 eval @ 4k every 10 rollouts.

Attention backends follow the GPU generation: FA3 on Hopper; flashinfer rollout and FA2 training
on Blackwell, where FA3 has no kernels.

Args:
    --hardware    auto (default) detects the GPU of the node the launcher runs on; pass H100/H200/B200/...
                  to override.

Usage:
    python3 scripts/run_qwen3_0_6b_fsdp.py
    python3 scripts/run_qwen3_0_6b_fsdp.py --hardware B200
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

HF_REPO = "Qwen/Qwen3-0.6B"
MODEL_NAME = "Qwen3-0.6B"
WANDB_GROUP = "qwen3-0.6B-fsdp-dapo4k"


@dataclass(frozen=True)
class _AttentionBackends:
    sglang: str
    train: str


# Keyed by command_utils.GENERATION_HARDWARE. FA3 kernels are built for Hopper (SM 9.0) only.
_ATTENTION_BACKENDS_OF_GENERATION = {
    "Hopper": _AttentionBackends(sglang="fa3", train="flash_attention_3"),
    "Blackwell": _AttentionBackends(sglang="flashinfer", train="flash_attention_2"),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_gpus_per_node: int = 4
    num_rollout: int = 100
    data_dir: str = "/root"
    model_dir: str = "/root/models"
    wandb_project: str = "miles-fsdp-curve"
    extra_args: str = ""
    hardware: Literal["auto", "H100", "H200", "B200", "B300", "GB200", "GB300"] = "auto"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    U = args.create_backend()
    model_path = f"{args.model_dir}/{MODEL_NAME}"
    # resolved here, not in __post_init__: prepare runs GPU-free and auto-detection needs a GPU
    attention = _ATTENTION_BACKENDS_OF_GENERATION[
        command_utils.GENERATION_HARDWARE[command_utils.resolve_hardware(args)]
    ]

    ckpt_args = f"--hf-checkpoint {model_path} " f"--ref-load {model_path} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 10 "
        f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 4096 "
        "--eval-top-p 1 "
    )

    grpo_args = (
        "--use-kl-loss "
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # No --wandb-key on purpose: exec_command_cpu prints the full command line, so the
    # trainer must pick up WANDB_API_KEY from its inherited environment instead.
    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {WANDB_GROUP} "
        if os.environ.get("WANDB_API_KEY")
        else ""
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-mem-fraction-static 0.75 "
        f"--sglang-attention-backend {attention.sglang} "
        "--sglang-chunked-prefill-size 4096 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--update-weight-buffer-size 536870912 "
        "--gradient-checkpointing "
        f"--attn-implementation {attention.train} "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} " f"--actor-num-gpus-per-node {args.num_gpus_per_node} " "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{eval_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{wandb_args} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{perf_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
