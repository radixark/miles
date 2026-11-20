"""Inkling 2 Mini full-parameter GRPO on one 8-GPU node with the torchtitan backend.

DAPO-Math-17k, 32 prompts x 8 samples, 4k responses. The trainer is FSDP8 x EP8 on BF16 weights dequantized
from the release checkpoint; the colocated TP1 SGLang engines serve the same BF16 weights.

Usage:
    python scripts/run_inkling2_mini.py
    python scripts/run_inkling2_mini.py --num-rollout 3
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils.legacy as U

HF_REPO = "thinkingmachines/Inkling-2-Mini"
MODEL_NAME = "Inkling-2-Mini"
WANDB_GROUP = "inkling2-mini-dapo4k"

MAX_RESPONSE_LEN = 4096
# covers the longest DAPO prompt plus the whole response
SEQ_LEN = MAX_RESPONSE_LEN + 2048


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_gpus_per_node: int = 8
    num_rollout: int = 100
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    wandb_project: str = "miles-inkling2-mini"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")
    if not os.path.exists(f"{args.model_dir}/{MODEL_NAME}-bf16/model.safetensors.index.json"):
        U.exec_command_cpu(
            "python tools/convert_inkling2_to_bf16.py "
            f"--model-dir {args.model_dir}/{MODEL_NAME} --save-dir {args.model_dir}/{MODEL_NAME}-bf16"
        )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    model_path = f"{args.model_dir}/{MODEL_NAME}-bf16"

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
        f"--rollout-max-response-len {MAX_RESPONSE_LEN} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
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

    # WANDB_API_KEY comes from the inherited environment so the printed command line never carries it
    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {WANDB_GROUP} "
        if os.environ.get("WANDB_API_KEY")
        else ""
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.55 "
        "--sglang-attention-backend fa4 "
        "--sglang-page-size 128 "
        "--sglang-bf16-gemm-backend torch "
        "--sglang-disable-radix-cache "
        "--sglang-decode-log-interval 1000 "
    )

    train_backend_args = (
        "--train-backend torchtitan "
        "--titan-model-name inkling2 "
        "--titan-model-flavor looped "
        f"--seq-length {SEQ_LEN} "
        f"--expert-model-parallel-size {args.num_gpus_per_node} "
        "--micro-batch-size 1 "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
    )

    misc_args = (
        f"--hf-checkpoint {model_path} "
        "--model-name inkling2 "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{rollout_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{wandb_args} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        extra_env_vars={"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
