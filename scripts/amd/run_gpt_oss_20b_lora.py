"""gpt-oss-20b GRPO LoRA training script for AMD (MI350X / MI355X).

The checkpoint is loaded directly through Megatron-Bridge. ROCm uses Triton for
SGLang attention, MoE, and LoRA kernels.

Args:
  --hardware: MI350X or MI355X.
  --num-gpus-per-node: Override the default eight GPUs.
  --model-dir / --data-dir / --output-dir: Input and output directories.
  --wandb-team: W&B entity when the API key has no default entity.

Examples:
  python scripts/amd/run_gpt_oss_20b_lora.py prepare
  python scripts/amd/run_gpt_oss_20b_lora.py train --hardware MI355X
  python scripts/amd/run_gpt_oss_20b_lora.py full-train --hardware MI355X
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = "lmsys/gpt-oss-20b-bf16"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "gpt-oss-20b-bf16"
    megatron_model_type: str = "gpt-oss-20b"
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None

    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "gate_proj,up_proj,down_proj"
    lora_base_cpu_backup: bool = True

    # 8 prompts x 8 samples = one 64-sample optimizer step per rollout.
    num_rollout: int = 20
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 4096
    global_batch_size: int = 64
    lr: float = 1e-5

    rollout_num_gpus_per_engine: int = 4
    sglang_mem_fraction_static: float = 0.2

    enable_wandb: bool = True
    wandb_team: str | None = None
    extra_args: str = ""


def _set_rocm_environment() -> None:
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _resolve_num_gpus(args: ScriptArgs) -> tuple[str, int]:
    hardware = U.resolve_hardware(args)
    return hardware, args.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[hardware]


def _download_inputs(args: ScriptArgs) -> None:
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_HF_REPO} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


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
    print(f"[run] gpt-oss-20b LoRA on {hardware}: {num_gpus} GPUs, trainer TP=4, rollout TP=4")

    ckpt_args = f"--hf-checkpoint {args.model_dir}/{args.model_name} " "--megatron-to-hf-mode bridge "

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
        "--no-sglang-lora-use-virtual-experts "
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
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    grpo_args = "--advantage-estimator grpo " "--entropy-coef 0.00 " "--eps-clip 0.2 " "--eps-clip-high 0.28 "

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 4096 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-dtype bfloat16 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-max-lora-rank 32 "
        "--sglang-lora-backend triton "
        "--sglang-moe-runner-backend triton "
        "--sglang-attention-backend triton "
    )

    misc_args = (
        "--train-backend megatron "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--qkv-format bshd "
        "--attention-backend unfused "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {num_gpus} "
        f"--num-gpus-per-node {num_gpus} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args(args)} "
        f"{perf_args} "
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
    """Download the checkpoint and training dataset."""
    _download_inputs(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Run GRPO LoRA training using prepared inputs."""
    _execute(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download inputs and run GRPO LoRA training."""
    _download_inputs(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
