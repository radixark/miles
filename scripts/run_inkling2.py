"""
Inkling2 RL training script (Inkling2 and its 8-layer toy).

Supports:
  - Inkling2                   48-layer MoE (384 routed experts, top-8), released with
                               NVFP4/MXFP8 experts. LoRA on 8 SM103 GPUs (B300, or two
                               4-GPU GB300 nodes): the trainer shards experts over all 8
                               (~160 GB/GPU of bf16 base weights), each node runs its own
                               engine, and the offloaded trainer goes to local NVMe.
  - Inkling2-8layer-64e-bf16   first 8 layers and 64 experts, experts dequantized to bf16:
                               local and full attention plus an attention-residual block
                               boundary; full-parameter or LoRA on one 4- or 8-GPU node.
  - Inkling2-8layer-64e        the same slice at release precision: NVFP4 experts, and the
                               last two layers carry experts of the first MXFP8 layers;
                               the engine serves both encodings (SM100 / SM103).

Requires the HF checkpoint under --model-dir and its torch_dist conversion (`convert`). On
several nodes, every node must read the whole `<model>_torch_dist` (shared filesystem, or a
copy on each node).

Args:
  --model-name   one of the models above
  --train-mode   full | lora
  --model-dir    parent of the HF checkpoint and `<model>_torch_dist`
  --data-dir     parent of dapo-math-17k and geo3k
  --task         dapo_math (text) | geo3k (images through the frozen vision tower)

Examples:
  python scripts/run_inkling2.py convert --model-name Inkling2-8layer-64e-bf16
  python scripts/run_inkling2.py train --model-name Inkling2-8layer-64e-bf16 --train-mode full
  # two 4-GPU nodes already joined in one ray cluster
  MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head ip> python scripts/run_inkling2.py convert \
      --model-name Inkling2 --train-mode lora --num-nodes 2 --num-gpus-per-node 4
  MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head ip> python scripts/run_inkling2.py train \
      --model-name Inkling2 --train-mode lora --num-nodes 2 --num-gpus-per-node 4
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

_MODEL_TYPES = {
    "Inkling2": "inkling2",
    "Inkling2-8layer-64e-bf16": "inkling2-8layer-64e",
    "Inkling2-8layer-64e": "inkling2-8layer-64e",
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: Literal["Inkling2", "Inkling2-8layer-64e-bf16", "Inkling2-8layer-64e"] = "Inkling2-8layer-64e-bf16"
    train_mode: Literal["full", "lora"] = "full"
    task: Literal["dapo_math", "geo3k"] = "dapo_math"
    num_rollout: int = 100
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    rollout_max_response_len: int = 4096
    sglang_context_length: int = 8192
    lr: float | None = None
    lora_rank: int = 32
    lora_alpha: int = 32
    enable_r3: bool = True
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    extra_args: str = ""

    def __post_init__(self):
        if self.lr is None:
            self.lr = 2e-5 if self.train_mode == "lora" else 1e-6
        assert self.train_mode == "lora" or not self.is_full_size, "full-size Inkling2 is LoRA only"

    @property
    def is_full_size(self) -> bool:
        return self.model_name == "Inkling2"

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def megatron_model_type(self) -> str:
        return _MODEL_TYPES[self.model_name]


def _parallel_args(args: ScriptArgs) -> str:
    # Experts shard over every GPU of the job; attention runs TP1 (its K/V sconv and
    # attention-residual state are not sequence-sharded).
    return (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.num_nodes * args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
    )


def _sglang_args(args: ScriptArgs) -> str:
    gpus = args.num_gpus_per_node
    return (
        f"--rollout-num-gpus-per-engine {gpus} "
        f"--sglang-ep-size {gpus} "
        "--sglang-attention-backend fa4 "
        "--sglang-bf16-gemm-backend torch "
        "--sglang-page-size 128 "
        "--sglang-mem-fraction-static 0.6 "
        f"--sglang-context-length {args.sglang_context_length} "
    )


def _lora_args(args: ScriptArgs) -> str:
    if args.train_mode == "full":
        return ""
    return (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        "--target-modules all-linear "
        "--experts-shared-outer-loras "
        # The engines keep the base they loaded across pauses, so syncs carry adapters only: a
        # resynced NVFP4 base would come back requantized, its amax recomputed from the dequantized
        # weights instead of the checkpoint's (up to 0.4% per expert).
        "--lora-base-cpu-backup "
        "--sglang-lora-backend triton "
        "--sglang-max-loras-per-batch 1 "
        f"--sglang-max-lora-rank {args.lora_rank} "
    )


def _train_args(args: ScriptArgs) -> str:
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {args.hf_checkpoint}_torch_dist "
        "--megatron-to-hf-mode raw "
        "--no-load-optim --no-load-rng --finetune "
    )
    match args.task:
        case "dapo_math":
            task_args = f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --apply-chat-template "
        case "geo3k":
            # Prompts stay structured messages: the multimodal processor renders them with their images.
            task_args = (
                f"--prompt-data {args.data_dir}/geo3k/geo3k_train.jsonl "
                "--custom-model-provider-path miles_plugins.models.inkling2.model.inkling2_mm_model_provider "
                "--sglang-enable-multimodal "
            )
    rollout_args = (
        f"{task_args}"
        "--input-key prompt --label-key label "
        "--rollout-shuffle --rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    algorithm_args = "--advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 --use-tis "
    if args.enable_r3:
        algorithm_args += "--use-rollout-routing-replay "
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--use-distributed-optimizer --accumulate-allreduce-grads-in-fp32 "
    )
    perf_args = (
        f"{_parallel_args(args)} "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
    )
    if args.is_full_size:
        # The trainer's bf16 base would not fit in host RAM beside the engines' base backup.
        perf_args += "--offload-train-target disk "
    misc_args = (
        "--bf16 --attention-dropout 0.0 --hidden-dropout 0.0 --no-bias-dropout-fusion "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    return (
        f"{ckpt_args} {rollout_args} {algorithm_args} {optimizer_args} {perf_args} "
        f"{_lora_args(args)} {_sglang_args(args)} {misc_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} {args.extra_args}"
    )


@app.command()
@command_utils.dataclass_cli
def convert(args: ScriptArgs):
    """HF checkpoint -> torch_dist under --model-dir (skipped when already converted)."""
    args.create_backend().convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=args.num_nodes > 1,
        num_nodes=args.num_nodes,
        extra_args=_parallel_args(args),
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    args.create_backend().execute_train(
        train_args=_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            # Cross-node NVLink (MNNVL, GB300 pods) needs NCCL's cuMem buffers; miles defaults the
            # trainer to NCCL_CUMEM_ENABLE=0, which falls back to sockets between nodes.
            **({"NCCL_CUMEM_ENABLE": "1"} if args.num_nodes > 1 else {}),
        },
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
