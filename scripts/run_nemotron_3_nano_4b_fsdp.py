"""FSDP training-curve validation: NVIDIA-Nemotron-3-Nano-4B-BF16 (model_type=nemotron_h, smallest of
the series; dense Mamba2+attention hybrid), single node.

dapo-math-17k @ 4k response len, AIME-2024 eval @ 4k every 10 rollouts.

attn-implementation stays flash_attention_2: the nemotron_h packed-doc reset patch calls flash_attn
(FA2) varlen directly.

Usage:
    python3 scripts/run_nemotron_3_nano_4b_fsdp.py
"""

import json
import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

HF_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MODEL_NAME = "NVIDIA-Nemotron-3-Nano-4B-BF16"
WANDB_GROUP = "nemotron-3-nano-4B-fsdp-dapo4k"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_gpus_per_node: int = 4
    num_rollout: int = 100
    data_dir: str = "/root"
    model_dir: str = "/root/models"
    wandb_project: str = "miles-fsdp-curve"
    extra_args: str = ""


def _strip_remote_code(model_path: str):
    """Drop auto_map from the local checkpoint config so transformers loads its native nemotron_h
    classes: the checkpoint's bundled remote code predates FA2 support and rejects it."""
    config_path = f"{model_path}/config.json"
    with open(config_path) as f:
        config = json.load(f)
    if "auto_map" not in config:
        return
    os.rename(config_path, f"{config_path}.bak-automap")
    del config["auto_map"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Removed auto_map from {config_path} (backup: config.json.bak-automap)")


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
    _strip_remote_code(f"{args.model_dir}/{MODEL_NAME}")


def execute(args: ScriptArgs):
    model_path = f"{args.model_dir}/{MODEL_NAME}"

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
        "--rollout-num-gpus-per-engine 1 " "--sglang-decode-log-interval 1000 " "--sglang-mem-fraction-static 0.7 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--update-weight-buffer-size 536870912 "
        "--gradient-checkpointing "
        "--attn-implementation flash_attention_2 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
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
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
