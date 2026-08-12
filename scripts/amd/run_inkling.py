"""
Inkling training script for ROCm (Inkling-Small 4-layer slice).

Supports:
  - Inkling-Small-4layer  4-layer slice of Inkling-Small (fits the 4-GPU CI lane).

Usage patterns:

  1. Train on pre-staged checkpoints:
       python scripts/amd/run_inkling.py train \
           --model-name Inkling-Small-4layer --num-nodes 1 --num-gpus-per-node 4

  2. Individual steps (rsync shared -> node-local NVMe, then train):
       python scripts/amd/run_inkling.py prepare-cp --model-name Inkling-Small-4layer
       python scripts/amd/run_inkling.py train --model-name Inkling-Small-4layer ...

  3. One-shot (prepare-cp when torch_dist_local differs, then train):
       python scripts/amd/run_inkling.py full-train --model-name Inkling-Small-4layer ...
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# model name -> scripts/models/<type>.sh; the 4-layer slices reuse the base
# definition with MODEL_ARGS_NUM_LAYERS=4 (set in ScriptArgs.__post_init__)
_MODEL_REGISTRY = {
    "Inkling-Small-4layer": "inkling-small",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["Inkling-Small-4layer"] = "Inkling-Small-4layer"

    train_mode: Literal["full"] = "full"
    task: Literal["dapo_math"] = "dapo_math"
    enable_eval: bool = False
    num_rollout: int = 100
    rollout_batch_size: int = 32
    global_batch_size: int = 64

    hf_checkpoint: str | None = None
    torch_dist: str | None = None
    torch_dist_local: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    save_dir: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 4
    rollout_num_gpus_per_engine: int = 4
    lr: float | None = None
    rollout_max_response_len: int = 4096
    sglang_context_length: int = 8192
    train_offload_disk_dir: str = "/tmp/train_offload"
    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)

    enable_r3: bool = True

    extra_args: str = ""

    def __post_init__(self):
        if self.model_name.endswith("-4layer"):
            os.environ["MODEL_ARGS_NUM_LAYERS"] = "4"
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.torch_dist is None:
            self.torch_dist = f"{self.model_dir}/{self.model_name}_torch_dist"
        if self.torch_dist_local is None:
            self.torch_dist_local = self.torch_dist
        if self.lr is None:
            self.lr = 1e-6
        self.colocate = True
        self.actor_num_nodes = self.num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node


def _get_parallel_config(args: ScriptArgs) -> str:
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    total_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.model_name == "Inkling-Small-4layer" and args.actor_num_nodes == 1:
        return (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} GPUs. "
        f"Please specify your parallel config in `scripts/amd/run_inkling._get_parallel_config`."
    )


def _train(args: ScriptArgs):
    print(
        f"running {args.model_name} {args.train_mode}/{args.task} on "
        f"{args.num_nodes} nodes (colocate) x {args.num_gpus_per_node} GPUs"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {args.torch_dist_local} "
        "--model-name inkling "
        "--megatron-to-hf-mode raw "
        "--no-load-optim --no-load-rng --finetune "
    )
    if args.save_dir is not None:
        ckpt_args += f"--save {args.save_dir}/{args.run_id}/checkpoints --save-interval 10 "

    rollout_args = (
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--apply-chat-template "
    )
    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 5 "
            f"--eval-prompt-data aime25 {args.data_dir}/aime-2025/aime-2025.jsonl "
            "--eval-input-key prompt --eval-label-key label "
            "--n-samples-per-eval-prompt 1 --eval-temperature 1 "
            f"--eval-max-response-len {args.rollout_max_response_len} "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--eps-clip-c 3.0 "
        "--use-tis "
    )
    if args.enable_r3:
        grpo_args += "--use-rollout-routing-replay "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-distributed-optimizer "
        "--no-check-for-nan-in-loss-and-grad "
        "--accumulate-allreduce-grads-in-fp32 "
        "--offload-train-target disk "
        f"--offload-train-disk-dir {args.train_offload_disk_dir} "
    )

    perf_args = _get_parallel_config(args)
    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-max-running-requests 64 "
        "--sglang-max-total-tokens 327680 "
        "--sglang-attention-backend triton "
        "--sglang-moe-runner-backend triton "
        "--sglang-mamba-scheduler-strategy extra_buffer "
        "--sglang-enable-multimodal "
        f"--sglang-context-length {args.sglang_context_length} "
        "--sglang-disable-custom-all-reduce "
    )

    misc_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--no-bias-dropout-fusion "
        "--distributed-timeout-minutes 30 "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    extra_env_vars = {
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        "SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM": "false",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "MILES_SGLANG_DUMMY_LOAD": "0",
        "SGLANG_SERVER_ENGINE_ROLLOUT_RETURN_LOGPROB": "1",
        "RAY_memory_monitor_refresh_ms": "0",
        "NCCL_MNNVL_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_RAS_ENABLE": "0",
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MODEL_REGISTRY[args.model_name],
        train_script="train.py",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(path_src=args.torch_dist, path_dst=args.torch_dist_local)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    """Copy the shared torch_dist checkpoint to node-local NVMe (torch_dist_local)."""
    _prepare_cp(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    if args.torch_dist_local != args.torch_dist:
        _prepare_cp(args)
    _train(args)


if __name__ == "__main__":
    app()
