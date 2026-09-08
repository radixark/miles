"""Qwen3-8B GRPO LoRA training script for AMD (MI350X / MI355X).

=====================

The dense-model LoRA recipe, trained through the bridge path. On ROCm the rollout runs on
SGLang's triton attention and LoRA backends; fa3/fa4/flashinfer have no HIP kernels. Ray is
told not to blank HIP/CUDA visibility for the job entrypoint.

PP stays 1 so a single rank holds a complete adapter to push to the rollout engines. TP=1 is
a recipe choice: an 8B model fits on one MI350X, so the node is spent on data parallelism.

=====================

Args:
  --hardware: MI350X or MI355X, which fixes the default GPU count per node.
  --num-gpus-per-node: Override the GPU count, e.g. when only some devices are visible.
  --task: gsm8k or dapo-math; picks the prompt dataset and the default response length.
  --lora-rank / --lora-alpha / --target-modules: LoRA geometry.
  --enable-eval: Run AIME evaluation every 20 steps.
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/amd/run_qwen3_8b_lora.py prepare
  python scripts/amd/run_qwen3_8b_lora.py train --hardware MI350X
  python scripts/amd/run_qwen3_8b_lora.py full-train --task gsm8k --enable-eval
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = "Qwen/Qwen3-8B"

_DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-8B"
    megatron_model_type: str = "qwen3-8B"
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None
    task: Literal["gsm8k", "dapo-math"] = "dapo-math"

    save_interval: int = 10
    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    # off gives fake on-policy under colocate: KL ~1.0 instead of ~1e-4
    lora_base_cpu_backup: bool = True

    # rollout
    num_rollout: int = 30
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 0  # 0 => per-task default
    global_batch_size: int = 128

    # rollout engine
    rollout_num_gpus_per_engine: int = 1
    sglang_mem_fraction_static: float = 0.5
    sglang_lora_backend: str = "triton"
    sglang_attention_backend: str = "triton"

    enable_eval: bool = False
    enable_wandb: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = 8192 if self.task == "dapo-math" else 512

    def resolve_gpus(self):
        """Not in __post_init__: detection needs a GPU, and `prepare` runs on a CPU node."""
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def _set_rocm_env():
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command_cpu(f"hf download {_HF_REPO} --local-dir {args.model_dir}/{args.model_name}")
    _download_dataset(args)


def _train(args: ScriptArgs):
    args.resolve_gpus()
    _set_rocm_env()
    print(
        f"[run] Qwen3-8B LoRA on {args.hardware}: rank={args.lora_rank}, "
        f"{args.num_gpus_per_node} GPUs, rollout tp={args.rollout_num_gpus_per_engine}"
    )
    load_save_path = f"{args.output_dir}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        "--megatron-to-hf-mode bridge "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {args.save_interval} "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
    )
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    match args.task:
        case "gsm8k":
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-dtype bfloat16 "
        "--sglang-decode-log-interval 1000 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
        f"--sglang-attention-backend {args.sglang_attention_backend} "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--update-weight-buffer-size 536870912 "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{wandb_args} {perf_args} {eval_args} {sglang_args} "
        f"{misc_args} {args.extra_args} "
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
def prepare(args: ScriptArgs):
    """Download the Qwen3-8B checkpoint and the task dataset. Run once per node before training."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training (assumes the dataset is already prepared)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download the checkpoint + dataset, then run GRPO LoRA training."""
    _prepare_download(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
