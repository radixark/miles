"""
Qwen3.5/3.6 35B-A3B GRPO LoRA training script (Megatron-Bridge / bridge mode).

Qwen3.5-35B-A3B is a hybrid MoE: within every ``full_attention_interval`` (=4)
decoder layers the last one is full softmax attention and the preceding three are
GDN (gated-delta-net) linear attention; every layer carries 256 routed experts +
1 shared expert. Qwen3.6-35B-A3B ships the same architecture and HF classes
(``Qwen3_5MoeForConditionalGeneration``), so both run through this script.

LoRA trains through the bridge path (``--megatron-to-hf-mode bridge``); the
registry ``.sh`` only satisfies megatron argparse and its ``--spec`` is inert.

Default target modules are wildcards anchored at ``language_model.decoder.layers.*``:
this keeps LoRA off the MTP block (``language_model.mtp.*`` has no adapter export
mapping, and SGLang cannot serve LoRA with MTP speculative decoding anyway) and off
the vision tower. The set covers attention (linear_qkv/linear_proj), MoE routed +
shared experts (linear_fc1/fc2), and the GDN projections: the fused megatron
``in_proj`` (q|k|v|z|b|a slices, served by SGLang as in_proj_qkvz + in_proj_ba)
and ``out_proj``.

Constraints baked into the parallel config:
  * TP <= 2: num_query_groups=2 caps tensor parallelism for the 35B-A3B geometry.
  * ``--qkv-format bshd``: megatron-core GatedDeltaNet rejects packed (thd) sequences.

Usage:
  python scripts/run_qwen3_5_35b_a3b_lora.py prepare    --model-name Qwen3.5-35B-A3B
  python scripts/run_qwen3_5_35b_a3b_lora.py full-train --model-name Qwen3.5-35B-A3B
  python scripts/run_qwen3_5_35b_a3b_lora.py train      --model-name Qwen3.6-35B-A3B --task dapo-math
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = {
    "Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
}

_MEGATRON_MODEL_TYPE = {
    "Qwen3.5-35B-A3B": "qwen3.5-35B-A3B_lora",
    "Qwen3.6-35B-A3B": "qwen3.6-35B-A3B_lora",
}

# Anchored below decoder.layers: keeps LoRA off the MTP block and the vision tower.
_LAYERS = "language_model.decoder.layers.*"
_DEFAULT_TARGET_MODULES = ",".join(
    [
        f"{_LAYERS}.self_attention.linear_qkv",
        f"{_LAYERS}.self_attention.linear_proj",
        f"{_LAYERS}.mlp.experts.linear_fc1",
        f"{_LAYERS}.mlp.experts.linear_fc2",
        f"{_LAYERS}.mlp.shared_experts.linear_fc1",
        f"{_LAYERS}.mlp.shared_experts.linear_fc2",
        f"{_LAYERS}.self_attention.in_proj",
        f"{_LAYERS}.self_attention.out_proj",
    ]
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal[
        "Qwen3.5-35B-A3B",
        "Qwen3.6-35B-A3B",
    ] = "Qwen3.5-35B-A3B"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # performance
    num_gpus_per_node: int = 8

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    # required for true on-policy under colocate (OFF -> KL ~1.0 vs ~1e-3)
    lora_base_cpu_backup: bool = True
    # MoE-expert LoRA layout: shared-outer when True, per-expert when False
    experts_shared_outer_loras: bool = True

    # rollout
    num_rollout: int = 10
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 0  # 0 => per-task default (gsm8k 512, dapo-math 4096)
    global_batch_size: int = 64

    # rollout engine
    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction_static: float = 0.4
    sglang_lora_backend: str = "triton"

    enable_wandb: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = 4096 if self.task == "dapo-math" else 512

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _get_parallel_config(args: ScriptArgs) -> str:
    """Single-node layout: TP2 (num_query_groups=2 caps TP), EP = num GPUs, DP for the rest.

    bshd is required because the megatron-core GatedDeltaNet forward rejects packed
    sequences; with --micro-batch-size 1 the batch stays unpacked.
    """
    return (
        "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--qkv-format bshd --micro-batch-size 1 --max-tokens-per-gpu 4096 "
    )


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.data_dir} {args.model_dir}")
    repo = _HF_REPO.get(args.model_name)
    if repo is not None:
        U.exec_command(f"hf download {repo} --local-dir {args.model_dir}/{args.model_name}")
    _download_dataset(args)


def _train(args: ScriptArgs):
    print(
        f"[run] Qwen3.5/3.6 LoRA: model={args.model_name} (megatron_model_type={args.megatron_model_type}), {args.num_gpus_per_node} GPUs, rollout tp={args.rollout_num_gpus_per_engine}"
    )
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "

    lora_args = f'--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} --target-modules "{args.target_modules}" '
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    lora_args += "--no-gradient-accumulation-fusion "
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
    )
    match args.task:
        case "gsm8k":  # zhuzilin/gsm8k ships {messages, label} parquet
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":  # zhuzilin/dapo-math-17k ships {prompt, label} jsonl (prompt = chat messages)
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    perf_args = _get_parallel_config(args)

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-dtype bfloat16 --sglang-decode-log-interval 1000 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
    )

    save_args = f"--save-interval 1 --save {load_save_path} "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--update-weight-buffer-size 536870912 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} {wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the model checkpoint and the task dataset (gsm8k or dapo-math). Run once per node before training."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training (assumes the dataset is already prepared)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download the model checkpoint + dataset, then run GRPO LoRA training."""
    _prepare_download(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
