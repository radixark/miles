"""Qwen3-4B GRPO LoRA training script for AMD (MI350X / MI355X).

This is the AMD counterpart of
``examples/lora/run-qwen3-4b-megatron-lora-result.sh``. It keeps the validated
Megatron-Bridge LoRA recipe and pins SGLang's attention and LoRA backends to Triton,
because the CUDA-only rollout backends are unavailable on ROCm.

The checkpoint is loaded directly from Hugging Face through Megatron-Bridge; no converted
Megatron checkpoint is required. ``full-train`` downloads the checkpoint and datasets
before submitting the training job.

Args:
  --hardware: MI350X or MI355X, which fixes the default GPU count per node.
  --num-gpus-per-node: Override the GPU count when only some devices are visible.
  --wandb-team: W&B entity (personal account or team); needed if the API key has no
    default entity.
  --model-dir / --data-dir / --output-dir: Model, dataset, and run output directories.
  --enable-eval: Evaluate AIME 2024 every 20 optimizer steps (default: on).

Examples:
  python scripts/amd/run_qwen3_4b_lora.py prepare
  python scripts/amd/run_qwen3_4b_lora.py train --hardware MI355X --wandb-team my-entity
  python scripts/amd/run_qwen3_4b_lora.py full-train --hardware MI355X --wandb-team my-entity
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = "Qwen/Qwen3-4B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: str = "qwen3-4B"
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None

    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    save_interval: int = 50

    # LoRA: inherited from the validated Qwen3-4B Megatron LoRA recipe.
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    lora_base_cpu_backup: bool = True

    # Rollout and optimizer shape: 8 prompts x 8 samples = one 64-sample update.
    num_rollout: int = 100
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 8192
    global_batch_size: int = 64
    lr: float = 2e-5

    # One 4B rollout engine per GPU.
    rollout_num_gpus_per_engine: int = 1
    sglang_mem_fraction_static: float = 0.4

    enable_eval: bool = True
    enable_wandb: bool = True
    wandb_team: str | None = None
    extra_args: str = ""


def _set_rocm_environment() -> None:
    """Preserve the caller's ROCm device selection through Ray startup."""
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    # Avoid execute_train's NVIDIA topology probe and disable unsupported NVLink SHARP.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _resolve_num_gpus(args: ScriptArgs) -> tuple[str, int]:
    hardware = U.resolve_hardware(args)
    num_gpus = args.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[hardware]
    return hardware, num_gpus


def _download_inputs(args: ScriptArgs) -> None:
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_HF_REPO} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _get_wandb_args(args: ScriptArgs) -> str:
    if not args.enable_wandb:
        return ""
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args and args.wandb_team:
        wandb_args += f"--wandb-team {shlex.quote(args.wandb_team)} "
    return wandb_args


def _execute(args: ScriptArgs) -> None:
    _set_rocm_environment()
    hardware, num_gpus = _resolve_num_gpus(args)
    print(f"[run] Qwen3-4B LoRA on {hardware}: {num_gpus} GPUs, rollout TP=1")

    checkpoint_dir = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        "--megatron-to-hf-mode bridge "
        f"--save {checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
    )
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime24 {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-decode-log-interval 1000 "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-lora-backend triton "
        "--sglang-attention-backend triton "
    )

    misc_args = (
        "--train-backend megatron "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--calculate-per-token-loss "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {num_gpus} "
        f"--num-gpus-per-node {num_gpus} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args(args)} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download Qwen3-4B and the configured training/evaluation datasets."""
    _download_inputs(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Run GRPO LoRA training using inputs already present on the node."""
    _execute(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download the inputs and run GRPO LoRA training."""
    _download_inputs(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
