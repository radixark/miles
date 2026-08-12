"""Qwen3 SFT training script.

=====================

One recipe covers both scales: Qwen3-4B-Base on a single node and the Qwen3-235B-A22B MoE
on four. They differ only in parallelism, the second Adam moment, whether the optimizer
state is offloaded to host RAM, and whether the launcher has to ssh the remaining nodes
into the ray cluster. Dataset, SFT rollout, learning-rate schedule and recompute settings
are shared.

This is pure SFT: `train_async.py` runs with `--debug-train-only`, so no SGLang engine is
started and there is neither generation nor eval. The checkpoint must already be converted
to Megatron `torch_dist`; this script only submits the training job.

=====================

Args:
  --model-name: Model variant, one of Qwen3-4B-Base / Qwen3-235B-A22B.
  --num-gpus-per-node: GPUs per node (default: 8).
  --join-ray-workers: For the multi-node recipe, ssh every host of /root/mpi_rack_hostfile
    into the ray cluster (default: on). Turn off when the cluster is already joined.
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_qwen3_sft.py --model-name Qwen3-4B-Base
  MASTER_ADDR=<head-ip> python scripts/run_qwen3_sft.py --model-name Qwen3-235B-A22B
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_MODEL_NAMES = Literal["Qwen3-4B-Base", "Qwen3-235B-A22B"]


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    actor_num_nodes: int
    tensor_model_parallel_size: int
    expert_model_parallel_size: int
    adam_beta2: float
    optimizer_cpu_offload: bool
    ssh_ray_workers: bool


_RECIPES: dict[str, _Recipe] = {
    # Qwen3-4B-Base is architecturally identical to Qwen3-4B, so it reuses that definition.
    "Qwen3-4B-Base": _Recipe("qwen3-4B", 1, 1, 1, 0.95, False, False),
    "Qwen3-235B-A22B": _Recipe("qwen3-235B-A22B", 4, 4, 32, 0.98, True, True),
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

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints "
        f"--save {args.output_dir}/checkpoints "
        "--save-interval 1000 "
    )

    sft_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.data_dir}/openhermes2_5.parquet "
        "--input-key messages "
        # no --apply-chat-template: sft_rollout renders the raw messages itself, together
        # with the per-token loss mask
        "--rollout-shuffle "
        "--num-epoch 3 "
        "--rollout-batch-size 128 "
        "--global-batch-size 128 "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        # no rollout generation at all, hence no sglang engine
        "--debug-train-only "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.recipe.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.recipe.expert_model_parallel_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style cosine "
        "--min-lr 1e-6 "
        "--lr-warmup-fraction 0.1 "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        f"--adam-beta2 {args.recipe.adam_beta2} "
    )
    if args.recipe.optimizer_cpu_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.recipe.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{sft_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
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
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
