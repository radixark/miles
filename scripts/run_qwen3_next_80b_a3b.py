"""Qwen3-Next-80B-A3B GSPO training script.

=====================

One recipe covers both topologies of the run. `4node` is the production layout: 4 nodes x
8 GPUs with the rollout engines colocated on the training GPUs, TP2/PP4/CP2/EP8, MTP
speculative decoding and DeepEP token dispatch. `single-node` fits the same model onto a
single 8-GPU node by dedicating 2 GPUs to rollout and 6 to training (PP6), which leaves
room for neither expert parallelism nor a draft model, so every batch dimension shrinks
along with it. Everything else -- dataset, GSPO constants, optimizer schedule, eval -- is
shared.

`4node` joins the remaining hosts to the local ray head over ssh, reading MASTER_ADDR and
MLP_WORKER_0_HOST from the environment; pass --no-join-ray-workers when the ray cluster
is already complete.

The checkpoint must already be converted to Megatron `torch_dist`; this script only
submits the training job.

=====================

Args:
  --topology: Cluster layout, one of 4node / single-node.
  --num-gpus-per-node: Physical GPUs per node (default: 8).
  --join-ray-workers: Join every host of /root/mpi_rack_hostfile to the ray cluster,
    multi-node topologies only (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_qwen3_next_80b_a3b.py --topology single-node
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_TOPOLOGIES = Literal["4node", "single-node"]


@dataclass(frozen=True)
class _Recipe:
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    # None means the rollout engines share the training GPUs
    rollout_num_gpus: int | None
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    rollout_temperature: float
    global_batch_size: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    max_tokens_per_gpu: int
    n_samples_per_eval_prompt: int
    eval_top_p: float
    rollout_num_gpus_per_engine: int
    sglang_ep_size: int
    enable_spec: bool
    moe_token_dispatcher_type: str
    moe_enable_deepep: bool


_RECIPES: dict[str, _Recipe] = {
    "4node": _Recipe(
        actor_num_nodes=4,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=None,
        num_rollout=3000,
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        rollout_temperature=1,
        global_batch_size=256,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=2,
        expert_model_parallel_size=8,
        max_tokens_per_gpu=8192,
        n_samples_per_eval_prompt=16,
        eval_top_p=1,
        rollout_num_gpus_per_engine=8,
        sglang_ep_size=8,
        enable_spec=True,
        moe_token_dispatcher_type="flex",
        moe_enable_deepep=True,
    ),
    "single-node": _Recipe(
        actor_num_nodes=1,
        actor_num_gpus_per_node=6,
        rollout_num_gpus=2,
        num_rollout=300,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        rollout_temperature=0.8,
        global_batch_size=64,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=6,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        max_tokens_per_gpu=2048,
        n_samples_per_eval_prompt=2,
        eval_top_p=0.7,
        rollout_num_gpus_per_engine=2,
        sglang_ep_size=1,
        enable_spec=False,
        moe_token_dispatcher_type="alltoall",
        moe_enable_deepep=False,
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    topology: _TOPOLOGIES = "4node"
    model_name: str = "Qwen3-Next-80B-A3B-Thinking"
    megatron_model_type: str = "qwen3-next-80B-A3B"
    num_gpus_per_node: int = 8
    join_ray_workers: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.topology]


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints "
        f"--save {args.output_dir}/checkpoints "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.recipe.num_rollout} "
        f"--rollout-batch-size {args.recipe.rollout_batch_size} "
        f"--n-samples-per-prompt {args.recipe.n_samples_per_prompt} "
        "--rollout-max-response-len 8192 "
        f"--rollout-temperature {args.recipe.rollout_temperature} "
        f"--global-batch-size {args.recipe.global_batch_size} "
        "--balance-data "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
        f"--n-samples-per-eval-prompt {args.recipe.n_samples_per_eval_prompt} "
        "--eval-max-response-len 16384 "
        f"--eval-top-p {args.recipe.eval_top_p} "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.recipe.tensor_model_parallel_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.recipe.pipeline_model_parallel_size} "
        f"--context-parallel-size {args.recipe.context_parallel_size} "
        f"--expert-model-parallel-size {args.recipe.expert_model_parallel_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.recipe.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
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

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.recipe.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.8 "
        f"--sglang-ep-size {args.recipe.sglang_ep_size} "
        "--sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 "
    )
    if args.recipe.enable_spec:
        sglang_args += (
            # mtp
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 2 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 3 "
            "--sglang-enable-draft-weights-cpu-backup "
            "--sglang-max-running-requests 512 "
        )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to drop this when using model with MLA
        "--attention-backend flash "
        f"--moe-token-dispatcher-type {args.recipe.moe_token_dispatcher_type} "
        f"--actor-num-nodes {args.recipe.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.recipe.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.recipe.moe_enable_deepep:
        misc_args += "--moe-enable-deepep "
    if args.recipe.rollout_num_gpus is None:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {args.recipe.rollout_num_gpus} "

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

    join_workers = args.join_ray_workers and args.recipe.actor_num_nodes > 1
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        before_ray_job_submit=(
            (
                lambda: U.ssh_start_ray_workers(
                    master_addr=os.environ["MASTER_ADDR"],
                    num_gpus_per_node=args.num_gpus_per_node,
                    head_host=os.environ.get("MLP_WORKER_0_HOST"),
                )
            )
            if join_workers
            else None
        ),
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
