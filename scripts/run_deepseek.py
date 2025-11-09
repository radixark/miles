import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))

import command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    model_name: str = "DeepSeek-V3"
    megatron_model_type: str = "deepseek-v3"
    num_gpus_per_node: int = 4
    enable_eval: bool = True
    extra_args: str = ""
    extra_env_vars: str = "{}"


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    """This script only needs to be executed on one node."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(
        f"huggingface-cli download deepseek-ai/{args.model_name} --local-dir /root/models/{args.model_name}"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    _fp8_cast_bf16(args)


def _fp8_cast_bf16(args: ScriptArgs):
    path_bf16_hf = f"/root/models/{args.model_name}-bf16/"
    if Path(path_bf16_hf).exists():
        return

    U.exec_command(
        "python tools/fp8_cast_bf16.py "
        f"--input-fp8-hf-path /root/models/{args.model_name} "
        f"--output-bf16-hf-path {path_bf16_hf}"
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _convert_to_megatron_ckpt(args)


def _convert_to_megatron_ckpt(args: ScriptArgs):
    """This script needs to be executed once per node."""
    path_dst = f"/root/models/{args.model_name}_torch_dist"
    if Path(path_dst).exists():
        return

    # `export SLURM_JOB_HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")`
    print(f"{os.environ.get('SLURM_JOB_HOSTNAMES')=} {os.environ.get('SLURM_NODEID')=}")
    master_addr = os.environ["SLURM_JOB_HOSTNAMES"].split("\n")[0]
    node_rank = int(os.environ["SLURM_NODEID"])
    U.exec_command(
        "source scripts/models/deepseek-v3.sh && "
        "PYTHONPATH=/root/Megatron-LM/ torchrun "
        f"--nproc-per-node {args.num_gpus_per_node} "
        f"--master-addr {master_addr} "
        "--master-port 23456 "
        f"--nnodes={args.num_nodes} "
        f"--node-rank {node_rank} "
        "tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--decoder-first-pipeline-num-layers 7 "
        "--decoder-last-pipeline-num-layers 6 "
        f"--hf-checkpoint /root/models/{args.model_name}-bf16/ "
        f"--save {path_dst} "
    )


# TODO improve these commadns
@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    _cp_model_to_local(args)


def _cp_model_to_local(args: ScriptArgs):
    path_src = f"/root/models/{args.model_name}_torch_dist"
    path_dst = f"/root/local_data/{args.model_name}_torch_dist"
    if Path(path_dst).exists():
        return

    U.exec_command(f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}")


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    run_id = U.create_run_id()

    load_save_path = f"/root/shared_data/{run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint /root/models/{args.model_name} "
        f"--ref-load /root/local_data/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
        "--rollout-temperature 0.8 "
        # ------------
        "--over-sampling-batch-size 256 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        # ------------
        "--num-steps-per-rollout 4 "
        "--balance-data "
    )

    # sometimes disable eval to speed up debugging
    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 8 "
            "--eval-max-response-len 32768 "
            "--eval-top-p 0.7 "
        )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--context-parallel-size 4 "
        "--expert-model-parallel-size 32 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 13 "
        # ------------
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # TODO run-deepseek-r1.sh enables use-kl-loss but w/ coef 0. can we just disable it like this?
        # "--use-kl-loss "
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
        # ------------
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 64 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tp-size 64 "
        "--sglang-ep-size 64 "
        # dp attention
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 8 "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-disable-radix-cache "
        # enable deepep for sglang
        "--sglang-enable-deepep-moe "
        "--sglang-deepep-mode auto "
        # make every dp rank has 128 concurrency
        "--sglang-server-concurrency 1024 "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        # use deepep for megatron
        "--moe-enable-deepep "
        "--moe-token-dispatcher-type flex "
        # ------------
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details /root/shared_data/{run_id}/dump_details "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        # TODO may get it from `config`
        num_gpus=args.num_gpus_per_node,
        model_type=args.megatron_model_type,
        extra_env_vars={**json.loads(args.extra_env_vars)},
    )


if __name__ == "__main__":
    app()
