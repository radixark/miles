"""GLM-4.5-355B-A32B GSPO training script for a rack of 8 nodes x 8 GPUs.

=====================

One recipe: the rollout engines are colocated on the training GPUs, Megatron runs
TP8 / PP4 / CP2 / EP16 with DeepEP token dispatch, and SGLang serves 32-GPU engines with
DP-attention and EAGLE MTP speculative decoding. Rollout is dynamically sampled -- 256
prompts oversampled down to a 128-prompt batch by reward standard deviation -- and every
rollout is replayed over 4 optimizer steps.

`scripts/run_glm45_355b_a32b.py` is the Blackwell / GRPO variant of the same model: a
smaller parallel layout, no dynamic sampling and its own checkpoint-preparation steps.

The shell script this replaces never checkpointed -- no --load, no --save -- so neither
does this launcher; --save turns writing `{output_dir}/checkpoints` every 20 steps back
on. The checkpoint must already be converted to Megatron `torch_dist`; this script only
submits the training job.

MASTER_ADDR must be exported: it is the ray head address, and every host of
--ray-hostfile is joined to it over ssh, skipping MLP_WORKER_0_HOST. The InfiniBand and
OpenMPI tuning this rack needs reaches the ray runtime env, the socket-interface part of
it only when MLP_SOCKET_IFNAME is set.

=====================

Args:
  --num-gpus-per-node: GPUs per node (default: 8).
  --actor-num-nodes: Nodes running Megatron (default: 8).
  --save: Write checkpoints to --output-dir every 20 steps (default: off, as in the .sh).
  --join-ray-workers: ssh the hosts of --ray-hostfile into the ray cluster (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  MASTER_ADDR=<head ip> python scripts/run_glm45_355b_a32b_8node.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "GLM-4.5-355B-A32B"
    megatron_model_type: str = "glm4.5-355B-A32B"
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 8
    save: bool = False
    join_ray_workers: bool = True
    ray_hostfile: str = "/root/mpi_rack_hostfile"
    num_rollout: int = 3000
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"


# InfiniBand / NVLink / OpenMPI tuning this rack needs, propagated by ray to every worker.
def _cluster_env_vars(master_addr: str) -> dict[str, str]:
    env_vars = {
        # execute_train exempts only 127.0.0.1 and the head; this cluster also needs 0.0.0.0
        "no_proxy": f"localhost,127.0.0.1,0.0.0.0,{master_addr}",
        "NCCL_CUMEM_ENABLE": "0",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
        "NCCL_IB_TC": "160",
        "NCCL_PXN_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_NET_GDR_LEVEL": "4",
        "NCCL_IB_RETRY_CNT": "7",
        "NCCL_IB_TIMEOUT": "32",
        "NCCL_IB_QPS_PER_CONNECTION": "8",
        "NCCL_P2P_LEVEL": "NVL",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        # off regardless of what execute_train's nvlink probe finds
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_CTAS": "4",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "^openib",
        "OMPI_MCA_routed": "direct",
        "OMPI_MCA_routed_radix": "1024",
        "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
    }
    # The shell script left ${MLP_SOCKET_IFNAME} for ray to interpolate; here an unset
    # variable would pin every socket to a literal empty interface name, so drop the keys.
    if (ifname := os.environ.get("MLP_SOCKET_IFNAME")) is not None:
        env_vars |= {
            "GLOO_SOCKET_IFNAME": ifname,
            "TP_SOCKET_IFNAME": ifname,
            "OMPI_MCA_oob_tcp_if_include": ifname,
            "OMPI_MCA_btl_tcp_if_include": ifname,
        }
    return env_vars


def execute(args: ScriptArgs):
    master_addr = os.environ.get("MASTER_ADDR")
    assert master_addr, "MASTER_ADDR is not set. Point it at the ray head (the .sh used $MLP_WORKER_0_HOST)."

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
    )
    if args.save:
        ckpt_args += f"--save {args.output_dir}/checkpoints " "--save-interval 20 "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 256 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--num-steps-per-rollout 4 "
        "--balance-data "
        "--rollout-stop-token-ids 151329 151336 151338 "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 8 "
        "--eval-max-response-len 32768 "
        "--eval-top-p 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 16 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    gspo_args = (
        "--advantage-estimator gspo "
        # --use-kl-loss stays off, so the reference model is never loaded and both KL
        # coefficients are inert; they are kept at 0.00 as the shell script had them.
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 1e-4 "
        "--eps-clip-high 2e-4 "
        "--use-tis "
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
        "--sglang-moe-dense-tp-size 1 "
        # mtp
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 1 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 2 "
        "--sglang-enable-draft-weights-cpu-backup "
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
        # overrides the alltoall dispatcher the model args set
        "--moe-token-dispatcher-type flex "
        "--moe-enable-deepep "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{gspo_args} "
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
        extra_env_vars=_cluster_env_vars(master_addr),
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
