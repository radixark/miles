"""torchtitan training-curve validation: Qwen3-0.6B, single node.

Same recipe as scripts/run_qwen3_0_6b_fsdp.py -- dapo-math-17k @ 4k response
len, AIME-2024 eval @ 4k every 10 rollouts -- so the two backends' curves can be
read against each other on one model.

Usage:
    python3 scripts/run_qwen3_0_6b_torchtitan.py
    python3 scripts/run_qwen3_0_6b_torchtitan.py --tp-size 2
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

HF_REPO = "Qwen/Qwen3-0.6B"
MODEL_NAME = "Qwen3-0.6B"
WANDB_GROUP = "qwen3-0.6B-torchtitan-dapo4k"

# Rotary tables and the buffers pipeline stages exchange are sized from this, so
# it has to cover the longest prompt plus the whole response.
SEQ_LEN = 8192


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_gpus_per_node: int = 4
    num_rollout: int = 100
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    data_dir: str = "/root"
    model_dir: str = "/root/models"
    wandb_project: str = "miles-torchtitan-curve"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    model_path = f"{args.model_dir}/{MODEL_NAME}"

    ckpt_args = f"--hf-checkpoint {model_path} --ref-load {model_path} "

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
        "--rollout-num-gpus-per-engine 1 " "--sglang-decode-log-interval 1000 " "--sglang-mem-fraction-static 0.75 "
    )

    # These are torchtitan's own parallelism field names; the FSDP shard degree
    # is deliberately absent, since torchtitan infers it from what the others
    # leave over. Note that qwen3-0.6B is weight-tied, so torchtitan refuses to
    # pipeline it -- --pp-size needs one of the untied flavors (8B and up).
    train_backend_args = (
        "--train-backend torchtitan "
        "--titan-model-name qwen3 "
        "--titan-model-flavor 0.6B "
        f"--titan-seq-len {SEQ_LEN} "
        f"--titan-tensor-parallel-degree {args.tp_size} "
        f"--titan-pipeline-parallel-degree {args.pp_size} "
        f"--titan-context-parallel-degree {args.cp_size} "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

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
