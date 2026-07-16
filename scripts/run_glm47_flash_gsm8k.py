"""GLM-4.7-Flash reasoning RL on GSM8K (single 8xH200 node).

Non-agentic math GRPO: rollouts are generated in-process by SGLang and scored
by the exact-match math reward (`--rm-type math`). No env server / agent server.

Usage (single node, 8 GPUs):
    python scripts/run_glm47_flash_gsm8k.py

Reuse a pre-staged checkpoint / dataset (skips download + conversion):
    python scripts/run_glm47_flash_gsm8k.py \
        --model-dir /shared/models --skip-prepare \
        --wandb-project glm47-flash-gsm8k
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-4.7-Flash"
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    hardware: Literal["H200"] = "H200"
    enable_eval: bool = True
    skip_prepare: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    wandb_project: str = "glm47-flash-gsm8k"


def prepare(args: ScriptArgs):
    if args.skip_prepare:
        print("prepare: --skip-prepare set, skipping model/dataset download + conversion")
        return

    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")

    hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    if Path(hf_checkpoint).exists():
        print(f"prepare: {hf_checkpoint} already exists, skipping HF download")
    else:
        U.exec_command(
            f"hf download {args.model_org}/{args.model_name} " f"--local-dir {hf_checkpoint}"
        )

    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def wandb_args(args: ScriptArgs) -> str:
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""
    return (
        "--use-wandb "
        f"--wandb-project {args.wandb_project} "
        f"--wandb-group {args.run_id} "
        f"--wandb-key '{wandb_key}' "
        "--disable-wandb-random-suffix "
    )


def execute(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 50} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 50} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {5 if args.mode == 'debug_minimal' else 300} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {256 if args.mode == 'debug_minimal' else 1024} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    # tp=4 because GLM-4.7-Flash has 20 attention heads (tp must divide num_heads)
    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.7 "
        # EAGLE speculative decoding (MTP)
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        # rollout routing replay (kept); NOTE: miles router intentionally NOT used
        "--use-rollout-routing-replay "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args(args)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
