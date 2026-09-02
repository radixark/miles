"""Qwen3 and Qwen3.6 SFT training script.

=====================

One recipe family covers Qwen3-4B-Base on a single node, the Qwen3-235B-A22B MoE
on four, and Qwen3.6-35B-A3B on a single 8-GPU H200 node. They share the SFT
rollout and optimizer schedule while keeping model-specific parallelism, loss
masking, MTP, and observability settings in the recipe table.

This is pure SFT: `train_async.py` runs with `--debug-train-only`, so no SGLang engine is
started and there is neither generation nor eval. The checkpoint must already be converted
to Megatron `torch_dist`; this script only submits the training job.

=====================

Args:
  --model-name: Model variant, one of Qwen3-4B-Base / Qwen3-235B-A22B /
    Qwen3.6-35B-A3B.
  --prompt-data: JSONL or Parquet dataset path. Defaults to
    <data-dir>/openhermes2_5.parquet.
  --input-key: Column containing a list of role/content messages (default: messages).
  --tools-key / --metadata-key: Columns containing tool definitions and row metadata.
  --checkpoint-dir: Training checkpoint directory. Defaults to <output-dir>/checkpoints.
  --trace-dir: Miles details and dashboard directory. Qwen3.6 enables observability
    and defaults this to <output-dir>/<run-id>/dump_details.
  --log-probs-chunk-size: Response-token chunk size for memory-efficient log-probability
    and entropy computation. Defaults to 4096 for Qwen3.6 and disabled for other recipes.
  --empty-unused-memory-level: Release cached CUDA blocks around the optimizer step.
    Level 2 also clears after gradient cleanup, before the next training step.
  --num-gpus-per-node: GPUs per node (default: 8).
  --join-ray-workers: For the multi-node recipe, ssh every host of /root/mpi_rack_hostfile
    into the ray cluster (default: on). Turn off when the cluster is already joined.
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_qwen3_sft.py --model-name Qwen3-4B-Base
  MASTER_ADDR=<head-ip> python scripts/run_qwen3_sft.py --model-name Qwen3-235B-A22B
  python scripts/run_qwen3_sft.py --model-name Qwen3.6-35B-A3B \
    --prompt-data /root/datasets/train.parquet \
    --checkpoint-dir /root/shared_data/qwen36-sft/checkpoints \
    --trace-dir /scratch/qwen36-sft/traces
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_MODEL_NAMES = Literal["Qwen3-4B-Base", "Qwen3-235B-A22B", "Qwen3.6-35B-A3B"]
_OPTIMIZERS = Literal["adam", "muon"]
_MUON_TP_MODES = Literal["blockwise", "duplicated", "distributed"]
_EMPTY_UNUSED_MEMORY_LEVELS = Literal[0, 1, 2]


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    actor_num_nodes: int
    tensor_model_parallel_size: int
    expert_model_parallel_size: int
    adam_beta2: float
    optimizer_cpu_offload: bool
    ssh_ray_workers: bool
    max_tokens_per_gpu: int = 9216
    loss_mask_type: str = "qwen"
    train_mtp: bool = False
    moe_token_dispatcher_type: str | None = None
    enable_observability: bool = False
    log_probs_chunk_size: int = -1
    recompute_loss_function: bool = False


_RECIPES: dict[str, _Recipe] = {
    # Qwen3-4B-Base is architecturally identical to Qwen3-4B, so it reuses that definition.
    "Qwen3-4B-Base": _Recipe("qwen3-4B", 1, 1, 1, 0.95, False, False),
    "Qwen3-235B-A22B": _Recipe("qwen3-235B-A22B", 4, 4, 32, 0.98, True, True),
    "Qwen3.6-35B-A3B": _Recipe(
        "qwen3.6-35B-A3B",
        1,
        1,
        8,
        0.98,
        True,
        False,
        max_tokens_per_gpu=8192,
        loss_mask_type="qwen3",
        train_mtp=True,
        moe_token_dispatcher_type="flex",
        enable_observability=True,
        log_probs_chunk_size=4096,
        recompute_loss_function=True,
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: _MODEL_NAMES = "Qwen3-4B-Base"
    num_gpus_per_node: int = 8
    join_ray_workers: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    prompt_data: str | None = None
    input_key: str = "messages"
    tools_key: str = "tools"
    metadata_key: str = "metadata"
    checkpoint_dir: str | None = None
    trace_dir: str | None = None
    save_interval: int = 1000
    num_epoch: int = 3
    rollout_batch_size: int = 128
    global_batch_size: int = 128
    max_tokens_per_gpu: int | None = None
    log_probs_chunk_size: int | None = None
    tensor_model_parallel_size: int | None = None
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int | None = None
    expert_tensor_parallel_size: int = 1
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    optimizer: _OPTIMIZERS = "adam"
    grad_reduce_in_bf16: bool = False
    muon_momentum: float = 0.95
    muon_extra_scale_factor: float = 0.2
    muon_tp_mode: _MUON_TP_MODES = "blockwise"
    muon_state_offload_chunk_size_mb: int = 256
    empty_unused_memory_level: _EMPTY_UNUSED_MEMORY_LEVELS = 0
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]

    @property
    def prompt_data_path(self) -> str:
        return self.prompt_data or f"{self.data_dir}/openhermes2_5.parquet"

    @property
    def checkpoint_path(self) -> str:
        return self.checkpoint_dir or f"{self.output_dir}/checkpoints"

    @property
    def details_path(self) -> str:
        return self.trace_dir or f"{self.output_dir}/{self.run_id}/dump_details"

    @property
    def tokens_per_gpu(self) -> int:
        return self.max_tokens_per_gpu or self.recipe.max_tokens_per_gpu

    @property
    def effective_log_probs_chunk_size(self) -> int:
        if self.log_probs_chunk_size is not None:
            return self.log_probs_chunk_size
        return self.recipe.log_probs_chunk_size

    @property
    def tensor_parallel_size(self) -> int:
        return self.tensor_model_parallel_size or self.recipe.tensor_model_parallel_size

    @property
    def expert_parallel_size(self) -> int:
        return self.expert_model_parallel_size or self.recipe.expert_model_parallel_size

    @property
    def observability_name(self) -> str:
        return self.wandb_run_name or self.run_id


def _validate_parallelism(args: ScriptArgs) -> None:
    sizes = {
        "tensor_model_parallel_size": args.tensor_parallel_size,
        "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
        "context_parallel_size": args.context_parallel_size,
        "expert_model_parallel_size": args.expert_parallel_size,
        "expert_tensor_parallel_size": args.expert_tensor_parallel_size,
    }
    for name, size in sizes.items():
        if size <= 0:
            raise ValueError(f"{name} must be positive, got {size}")

    world_size = args.recipe.actor_num_nodes * args.num_gpus_per_node
    decoder_parallel_size = args.tensor_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    expert_parallel_size = (
        args.expert_tensor_parallel_size * args.expert_parallel_size * args.pipeline_model_parallel_size
    )
    if world_size % decoder_parallel_size:
        raise ValueError(f"world_size={world_size} must be divisible by TP*PP*CP={decoder_parallel_size}")
    if world_size % expert_parallel_size:
        raise ValueError(f"world_size={world_size} must be divisible by ETP*EP*PP={expert_parallel_size}")
    if args.effective_log_probs_chunk_size == 0 or args.effective_log_probs_chunk_size < -1:
        raise ValueError(
            "log_probs_chunk_size must be -1 (disabled) or a positive integer, "
            f"got {args.effective_log_probs_chunk_size}"
        )
    if args.optimizer == "muon":
        if not 0.0 < args.muon_momentum < 1.0:
            raise ValueError(f"muon_momentum must be between 0 and 1, got {args.muon_momentum}")
        if args.muon_extra_scale_factor <= 0.0:
            raise ValueError(f"muon_extra_scale_factor must be positive, got {args.muon_extra_scale_factor}")
        if args.muon_state_offload_chunk_size_mb <= 0:
            raise ValueError(
                "muon_state_offload_chunk_size_mb must be positive, " f"got {args.muon_state_offload_chunk_size_mb}"
            )


def execute(args: ScriptArgs) -> None:
    _validate_parallelism(args)
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.checkpoint_path} "
        f"--save {args.checkpoint_path} "
        f"--save-interval {args.save_interval} "
    )

    sft_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.prompt_data_path} "
        f"--input-key {args.input_key} "
        f"--tool-key {args.tools_key} "
        f"--metadata-key {args.metadata_key} "
        # no --apply-chat-template: sft_rollout renders the raw messages itself, together
        # with the per-token loss mask
        "--rollout-shuffle "
        f"--num-epoch {args.num_epoch} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--global-batch-size {args.global_batch_size} "
        f"--loss-mask-type {args.recipe.loss_mask_type} "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        # no rollout generation at all, hence no sglang engine
        "--debug-train-only "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        f"--context-parallel-size {args.context_parallel_size} "
        f"--expert-model-parallel-size {args.expert_parallel_size} "
        f"--expert-tensor-parallel-size {args.expert_tensor_parallel_size} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.tokens_per_gpu} "
        f"--empty-unused-memory-level {args.empty_unused_memory_level} "
    )
    if args.effective_log_probs_chunk_size > 0:
        perf_args += f"--log-probs-chunk-size {args.effective_log_probs_chunk_size} "
    if args.recipe.recompute_loss_function:
        perf_args += "--recompute-loss-function "

    optimizer_name = "dist_muon" if args.optimizer == "muon" else "adam"
    optimizer_args = (
        f"--optimizer {optimizer_name} "
        f"--lr {args.learning_rate} "
        "--lr-decay-style cosine "
        f"--min-lr {args.min_learning_rate} "
        "--lr-warmup-fraction 0.1 "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        f"--adam-beta2 {args.recipe.adam_beta2}"
    )
    if args.optimizer == "muon":
        optimizer_args += (
            f" --muon-momentum {args.muon_momentum} "
            "--muon-nesterov "
            "--muon-scale-mode spectral "
            f"--muon-extra-scale-factor {args.muon_extra_scale_factor} "
            "--muon-coefficient-type quintic "
            "--muon-num-ns-steps 5 "
            f"--muon-tp-mode {args.muon_tp_mode} "
            "--chunked-optimizer-state-offload "
            "--optimizer-state-offload-fraction 1.0 "
            f"--optimizer-state-offload-chunk-size-mb {args.muon_state_offload_chunk_size_mb}"
        )
    elif args.recipe.optimizer_cpu_offload:
        optimizer_args += (
            " --optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer"
        )

    model_feature_args = ""
    if args.recipe.train_mtp:
        model_feature_args += "--enable-mtp-training --mtp-num-layers 1 --mtp-loss-scaling-factor 0.2 "
    if args.recipe.moe_token_dispatcher_type is not None:
        model_feature_args += f"--moe-token-dispatcher-type {args.recipe.moe_token_dispatcher_type} "

    gradient_reduction_args = (
        "--grad-reduce-in-bf16 " if args.grad_reduce_in_bf16 else "--accumulate-allreduce-grads-in-fp32 "
    )
    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        f"{gradient_reduction_args}"
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.recipe.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.recipe.enable_observability:
        misc_args += (
            "--observe-training-entropy "
            "--use-rollout-entropy "
            "--use-prometheus "
            f"--prometheus-run-name {args.observability_name} "
            "--use-miles-dashboard "
            "--dashboard-forward-prometheus "
            f"--dump-details {args.details_path} "
        )

    train_args = (
        f"{ckpt_args} "
        f"{sft_args} "
        f"{optimizer_args} "
        f"{model_feature_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.observability_name, project_name=args.wandb_project)} "
        f"{perf_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.recipe.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        before_ray_job_submit=(
            partial(
                U.ssh_start_ray_workers,
                master_addr=os.environ["MASTER_ADDR"],
                num_gpus_per_node=args.num_gpus_per_node,
                # under the MLP scheduler worker 0 is the ray head, which is already up
                head_host=os.environ.get("MLP_WORKER_0_HOST"),
            )
            if args.recipe.ssh_ray_workers and args.join_ray_workers
            else None
        ),
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
