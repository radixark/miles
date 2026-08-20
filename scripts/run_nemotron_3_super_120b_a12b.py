"""NVIDIA Nemotron-3-Super-120B-A12B GRPO training script (2 nodes x 8 GPUs).

=====================

nemotron_h is a hybrid Mamba2 + Attention + latent-MoE architecture (88 layers, 512
experts top-22, moe_latent_size=1024). It loads through NVIDIA ``megatron.bridge``
(``--megatron-to-hf-mode bridge``): ``--ref-load`` points at the HF checkpoint itself, so
no offline torch_dist conversion step is needed.

The two pods take different roles and each runs exactly one command:
  * `worker` waits for the head's ray port, joins the cluster and blocks. It never submits.
  * `train` starts the ray head, waits until the cluster reports every GPU of both nodes,
    and only then submits the job.

=====================

Args:
  --head-ip: IP of the head pod. Both roles need it.
  --num-nodes / --num-gpus-per-node: Cluster shape (default: 2 x 8).
  --num-rollout: Rollout steps (default: 10, i.e. the smoke test).
  --model-dir / --data-dir: Checkpoint / dataset directories. On the cluster these are
      /cluster_public/miles_data/models and /cluster_public/miles_data/datasets.

=====================

  on the worker pod:
    python scripts/run_nemotron_3_super_120b_a12b.py worker --head-ip <head_pod_ip>
  on the head pod:
    python scripts/run_nemotron_3_super_120b_a12b.py train --head-ip <head_pod_ip>
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    head_ip: str = "127.0.0.1"
    model_name: str = "NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    megatron_model_type: str = "nemotron-3-super-120b-a12b"
    num_nodes: int = 2
    num_gpus_per_node: int = 8
    num_rollout: int = 10
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"


def _wait_for_head_port(head_ip: str) -> None:
    for _ in range(60):
        # exec_command_cpu raises on a non-zero exit, so the probe reports through stdout.
        if U.exec_command_cpu(f"nc -z {head_ip} 6379 2>/dev/null; echo $?", capture_output=True).strip() == "0":
            return
        print(f"waiting for head {head_ip}:6379 ...")
        U.exec_command_cpu("sleep 5")


def _wait_for_ray_gpus(expected_gpus: int) -> None:
    """The head reports only its own 8 GPUs until the worker joins, and a job submitted then gets one node."""
    for _ in range(120):
        if f"{expected_gpus}.0 GPU" in U.exec_command_cpu("ray status 2>/dev/null || true", capture_output=True):
            print(f"[ray] cluster ready: {expected_gpus} GPUs")
            break
        U.exec_command_cpu("sleep 5")
    # Submit either way: the job's own resource error names the shortfall, and this leaves it in the log.
    U.exec_command_cpu("ray status")


def _execute_train(args: ScriptArgs):
    total_gpus = args.num_nodes * args.num_gpus_per_node
    hf_checkpoint = f"{args.model_dir}/{args.model_name}"

    ckpt_args = (
        # --ref-load at an HF directory routes load_checkpoint through megatron.bridge,
        # so the run needs no torch_dist copy of the weights.
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load {hf_checkpoint} "
        f"--save {args.output_dir}/checkpoints "
        "--save-interval 20 "
        "--megatron-to-hf-mode bridge "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 128 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 1024 "
        "--log-probs-chunk-size 128 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
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
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        "--sglang-mem-fraction-static 0.7 "
        # Replay the exact rollout routing during the training forward so train logprobs
        # match rollout logprobs (needed for MoE).
        "--use-rollout-routing-replay "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend auto "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {total_gpus} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        before_ray_job_submit=lambda: _wait_for_ray_gpus(total_gpus),
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Head role: start the ray head, wait for both nodes' GPUs, submit the job."""
    # execute_train reads MASTER_ADDR for the ray head's node ip and for the torch
    # distributed rendezvous the worker's ranks connect to.
    os.environ["MASTER_ADDR"] = args.head_ip
    _execute_train(args)


@app.command()
@U.dataclass_cli
def worker(args: ScriptArgs):
    """Worker role: join the head's ray cluster and block."""
    # A re-run inherits the previous run's agents, and ray refuses to join with them alive.
    U.exec_command_cpu("pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 miles; sleep 3; true; ")
    _wait_for_head_port(args.head_ip)
    U.exec_command_cpu(
        f"ray start --address={args.head_ip}:6379 "
        f"--num-gpus={args.num_gpus_per_node} "
        "--disable-usage-stats "
        "--block"
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
