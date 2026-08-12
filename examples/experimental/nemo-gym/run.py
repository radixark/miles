"""NeMo Gym launcher (Qwen3-4B-Instruct-2507): Miles <-> mini_swe_agent_2 orchestration.

Defaults are the exact configuration of the validated smoke run (4x H200,
2026-07-28): 3 GRPO steps at tiny scale against a NeMo Gym server running the
docker sandbox provider. Scale up --num-rollout / batch sizes for real
training.

Usage:
    NEMO_GYM_URL=http://<nemo-gym-host>:12000 python run.py
    python run.py --mode debug_rollout_only
    python run.py --skip-prepare --prompt-data /my/data.jsonl
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    megatron_model_type: str = "qwen3-4B-Instruct-2507"
    num_gpus_per_node: int = 4
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "Qwen3-4B-Instruct-2507"
    hf_checkpoint: str = "Qwen/Qwen3-4B-Instruct-2507"
    ref_load: str = "/root/Qwen3-4B-Instruct-2507_torch_dist"
    save_dir: str = "/root/Qwen3-4B-Instruct-2507_nemogym/"
    prompt_data: str = "/root/swe_verified.jsonl"

    # Training settings (validated smoke scale)
    max_seq_len: int = 16384
    rollout_max_response_len: int = 4096
    num_rollout: int = 3
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    save_interval: int = 1000

    # NeMo Gym settings
    nemo_gym_url: str = os.environ.get("NEMO_GYM_URL", "http://localhost:12000")
    # Trainer address reachable from the NeMo Gym host; only needed when that
    # host cannot resolve the trainer's hostname (e.g. it dials back over a
    # tailnet).
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)


def prepare(args: ScriptArgs):
    """Convert the HF checkpoint to torch_dist format if not already done."""
    U = args.create_backend()
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
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
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_seq_len} "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path nemogym_agent_function.run "
        "--custom-rm-path nemogym_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--use-session-server "
        # 0.0.0.0 so the NeMo Gym host can dial in on any interface (e.g. a
        # tailnet address); internal calls resolve it to localhost.
        "--session-server-ip 0.0.0.0 "
        "--session-server-port 30000 "
        "--session-server-workers 32 "
        "--tito-model qwen3 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
    )

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{grpo_args}"
        f"{optimizer_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{command_utils.repo_base_dir}",
        "NEMO_GYM_URL": args.nemo_gym_url,
    }
    if args.router_external_host:
        extra_env_vars["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
