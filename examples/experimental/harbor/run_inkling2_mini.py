"""Launcher: Inkling 2 Mini GRPO on Terminal-Bench 2 with Harbor in-process (terminus-2), torchtitan backend.

One 8-GPU node: the trainer is FSDP8 x EP8 on the BF16 checkpoint written by tools/convert_inkling2_to_bf16.py,
colocated with eight TP1 SGLang engines. Terminus-2 runs in the rollout worker and drives the task sandbox over the
network, so the sandboxes need no route back to the trainer.

Needs the radixark/miles:inkling-2-mini image (docker/Dockerfile.inkling2-mini): SGLang 1.0 with Inkling 2 Mini, and
Harbor with terminus-2 honoring max_seq_len.

Usage:
    HARBOR_ENV_TYPE=e2b python examples/experimental/harbor/run_inkling2_mini.py \
        --hf-checkpoint /root/models/Inkling-2-Mini-bf16 \
        --prompt-data /root/tb2_train.jsonl --harbor-tasks-dir /root/tb2
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer
from launch_common import (
    METADATA_KEY,
    agentic_pythonpath_dirs,
    agentic_train_args,
    harbor_env_vars,
    preflight_task_dirs,
)

import miles.utils.external_utils.command_utils.legacy as U
from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    num_gpus_per_node: int = 8

    hf_checkpoint: str = "/root/models/Inkling-2-Mini-bf16"
    prompt_data: str = "/root/tb2_train.jsonl"
    save_dir: str = ""
    save_interval: int = 20

    # a whole trajectory must fit one training sequence: no context parallelism on this model
    max_seq_len: int = 32768
    max_response_len: int = 8192
    num_rollout: int = 200
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    lr: float = 1e-6
    sglang_mem_fraction_static: float = 0.7
    save_traces_dir: str = ""

    # Harbor settings (see harbor_agent_function.py for what each does)
    harbor_env_type: str = os.environ.get("HARBOR_ENV_TYPE", "")
    harbor_env_kwargs: str = os.environ.get("HARBOR_ENV_KWARGS", "")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/tb2")
    harbor_trials_dir: str = os.environ.get("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    agent_timeout: int = int(os.environ.get("AGENT_TIMEOUT", "1800"))
    daytona_api_key_file: str = os.environ.get("DAYTONA_API_KEY_FILE", "")
    e2b_api_key_file: str = os.environ.get("E2B_API_KEY_FILE", "")
    modal_config_file: str = os.environ.get("MODAL_CONFIG_PATH", "")
    novita_api_key_file: str = os.environ.get("NOVITA_API_KEY_FILE", "")

    # prepended to the workers' PYTHONPATH, e.g. an SGLang checkout that serves Inkling 2 Mini
    extra_pythonpath: str = ""
    wandb_project: str = "miles-inkling2-mini-tb2"
    wandb_group: str = "tb2-terminus2"
    extra_args: str = ""


def execute(args: ScriptArgs):
    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        f"--metadata-key {METADATA_KEY} "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1 "
        f"--rollout-max-response-len {args.max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-attention-backend fa4 "
        "--sglang-page-size 128 "
        "--sglang-bf16-gemm-backend torch "
        "--sglang-router-port 31000 "
    )
    train_backend_args = (
        "--train-backend torchtitan "
        "--titan-model-name inkling2 "
        "--titan-model-flavor looped "
        f"--seq-length {args.max_seq_len} "
        f"--expert-model-parallel-size {args.num_gpus_per_node} "
        "--micro-batch-size 1 "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
    )
    agent_args = agentic_train_args(tito_model="inkling", session_server_workers=32)
    ckpt_args = f"--save {args.save_dir} --save-interval {args.save_interval} " if args.save_dir else ""
    # WANDB_API_KEY comes from the inherited environment so the printed command line never carries it
    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.wandb_group} "
        if os.environ.get("WANDB_API_KEY")
        else ""
    )
    misc_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        "--model-name inkling2 "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    trace_args = f"--dump-details {args.save_traces_dir} " if args.save_traces_dir else ""

    preflight_task_dirs(args.prompt_data, args.harbor_tasks_dir)
    # the inherited PYTHONPATH carries the image's Harbor directory (docker/Dockerfile.inkling2-mini)
    pythonpath = [
        *filter(None, [args.extra_pythonpath]),
        *agentic_pythonpath_dirs(),
        str(command_utils.repo_base_dir),
        *filter(None, [os.environ.get("PYTHONPATH")]),
    ]
    U.execute_train(
        train_args=(
            f"{rollout_args}{grpo_args}{optimizer_args}{sglang_args}{train_backend_args}{agent_args}"
            f"{ckpt_args}{wandb_args}{misc_args}{debug_args}{trace_args}{args.extra_args} "
        ),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        extra_env_vars={
            "PYTHONPATH": ":".join(pythonpath),
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
            **harbor_env_vars(args),
            # end the trial where the trainer would truncate it, instead of rolling on past what it can learn from
            "HARBOR_MAX_SEQ_LEN": os.environ.get("HARBOR_MAX_SEQ_LEN") or str(args.max_seq_len),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
