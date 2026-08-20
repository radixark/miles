"""Qwen3-235B-A22B GSPO training script.

=====================

Training and rollout are disaggregated here: 8 actor nodes x 8 GPUs run Megatron
(TP4 / PP4 / CP2 / EP16) while a separate 64-GPU pool serves SGLang in 32-GPU engines with
DP-attention, so the rack needs 128 GPUs (16 nodes of 8) and `--colocate` is not passed.

Rollout weights are FP8 while the reference model stays BF16: `--hf-checkpoint` points at
the FP8 HF release and `--ref-load` at the BF16 Megatron `torch_dist` conversion, so both
directories must exist before launching. This script only submits the training job.

MASTER_ADDR must be exported; it is both the ray head address and the address the workers
join. By default the launcher sshes into every host of the hostfile to join them to the
cluster, mirroring the shell script it replaces.

=====================

Args:
  --num-gpus-per-node: GPUs per node (default: 8).
  --rollout-fp8: Roll out from the FP8 checkpoint instead of the BF16 one (default: on).
  --enable-eval: Run AIME evaluation every 20 steps (default: off, see eval_args).
  --join-ray-workers: ssh the hosts of --ray-hostfile into the ray cluster (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  MASTER_ADDR=<head ip> python scripts/run_qwen3_235b_a22b.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-235B-A22B"
    megatron_model_type: str = "qwen3-235B-A22B"
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 8
    rollout_num_gpus: int = 64
    rollout_fp8: bool = True
    enable_eval: bool = False
    join_ray_workers: bool = True
    ray_hostfile: str = "/root/mpi_rack_hostfile"
    num_rollout: int = 3000
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"


def execute(args: ScriptArgs):
    master_addr = os.environ.get("MASTER_ADDR")
    assert master_addr, "MASTER_ADDR is not set. Please set it to the master node address."

    hf_checkpoint = (
        f"{args.model_dir}/{args.model_name}-FP8" if args.rollout_fp8 else f"{args.model_dir}/{args.model_name}"
    )
    load_save_path = f"{args.output_dir}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
        "--balance-data "
    )

    # The shell script configured the eval dataset but left --eval-interval commented out,
    # so periodic eval never fired; --enable-eval turns the interval back on.
    eval_args = "--eval-interval 20 " if args.enable_eval else ""
    eval_args += (
        f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 16384 "
        "--eval-top-p 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 16 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 22 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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
        "--rollout-num-gpus-per-engine 32 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 4 "
        "--sglang-ep-size 32 "
        "--sglang-enable-dp-lm-head "
        # was `1 2 4 8 $(seq 16 8 256)` in shell
        "--sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 "
        "--sglang-moe-a2a-backend deepep "
        "--sglang-deepep-mode auto "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        f"--update-weight-buffer-size {4 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
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
        before_ray_job_submit=(
            (
                lambda: U.ssh_start_ray_workers(
                    master_addr=master_addr,
                    num_gpus_per_node=args.num_gpus_per_node,
                    hostfile=args.ray_hostfile,
                    head_host=os.environ.get("MLP_WORKER_0_HOST"),
                )
            )
            if args.join_ray_workers
            else None
        ),
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
