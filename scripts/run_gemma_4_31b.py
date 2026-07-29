"""
Gemma-4 31B-it dense GRPO training script (single-node 8x H200).

Dense (non-MoE) counterpart of run_gemma_4_26b_a4b.py: TP4/DP2, no expert
parallelism. Trained via the HF<->Megatron bridge (`--megatron-to-hf-mode
bridge`); the dense config is driven directly through Gemma4VLBridge, so there
is no LLM-view rewrite or offline conversion — `prepare` only downloads.
MODEL_ARGS come from scripts/models/gemma-4-31b-it.sh.

Requires the radixark/Megatron-Bridge gemma4-dense branch.

Single-node smoke test:
  python scripts/run_gemma_4_31b.py full-train --num-nodes 1
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "google"
    model_name: str = "gemma-4-31B-it"
    megatron_model_type: str = "gemma-4-31b-it"
    num_gpus_per_node: int = 8
    enable_eval: bool = False
    num_rollout: int = 3000
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    hardware: Literal["H200", "H100", "B200"] = "H200"

    def __post_init__(self):
        if self.num_nodes == 1:
            self.mode = "debug_minimal"


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _execute_train(args: ScriptArgs):
    ckpt = f"{args.model_dir}/{args.model_name}"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {ckpt} "
        f"--ref-load {ckpt} "
        "--megatron-to-hf-mode bridge "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type gemma_math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {256 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 512 "
    )

    # no KL: the ref-model copy doesn't fit alongside sglang in colocate
    grpo_args = "--advantage-estimator grpo " "--entropy-coef 0.00 " "--eps-clip 0.2 " "--eps-clip-high 0.28 "

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

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        # keep KV pool small to leave room for the dense train footprint
        "--sglang-mem-fraction-static 0.5 "
        # triton: Gemma-4 global head_dim=512 exceeds FlashAttention's 256 cap
        "--sglang-attention-backend triton "
        "--sglang-disable-custom-all-reduce "
        "--sglang-disable-cuda-graph "
        "--sglang-disable-overlap-schedule "
        "--sglang-disable-radix-cache "
        # keep resident: the offload path crashes during update_weights for this model
        "--no-offload-train "
        "--no-offload-rollout "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--no-gradient-accumulation-fusion "
        "--no-check-for-nan-in-loss-and-grad "
        "--attention-softmax-in-fp32 "
        "--attention-backend unfused "
        "--qkv-format bshd "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
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


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Full pipeline: download, train."""
    _prepare_download(args)
    _execute_train(args)


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download model and data."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
