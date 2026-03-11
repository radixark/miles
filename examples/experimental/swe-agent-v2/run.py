#!/usr/bin/env python3
"""Agent V2 launcher: Miles <-> Harbor agent orchestration.

Supports any task type (SWE-bench, Terminal-Bench, custom) via Harbor.
Replaces run.sh / run_debug.sh with a single configurable entry point.

Usage:
    python run.py                          # full training (default)
    python run.py --mode debug             # debug with smaller batch/rollout
    python run.py --mode debug --num-gpus 2
    python run.py --hf-checkpoint Qwen/Qwen3-4B --model-script qwen3-4B.sh
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MILES_ROOT = SCRIPT_DIR.parents[2]


def check_has_nvlink() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return "NV" in out and any(f"NV{i}" in out for i in range(1, 100))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def exec_command(cmd: str):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=False)


def build_args(args: argparse.Namespace) -> str:
    is_debug = args.mode == "debug"

    model_script = MILES_ROOT / "scripts" / "models" / args.model_script
    if not model_script.exists():
        sys.exit(f"Model script not found: {model_script}")

    ckpt = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )
    if args.load_dir:
        ckpt += f"--load {args.load_dir} "

    perf = (
        f"--tensor-model-parallel-size {args.tp} "
        f"--pipeline-model-parallel-size {args.pp} "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    num_rollout = 50 if is_debug else args.num_rollout
    rollout_batch = 4 if is_debug else args.rollout_batch_size
    n_samples = 4 if is_debug else args.n_samples_per_prompt
    max_resp_len = 4096 if is_debug else args.rollout_max_response_len
    global_batch = 16 if is_debug else args.global_batch_size

    rollout = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch} "
        f"--n-samples-per-prompt {n_samples} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-max-response-len {max_resp_len} "
        f"--global-batch-size {global_batch} "
        "--balance-data "
    )

    grpo = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        f"--kl-loss-coef {args.kl_loss_coef} "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        f"--eps-clip {args.eps_clip} "
        f"--eps-clip-high {args.eps_clip_high} "
    )

    optimizer = (
        f"--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    wandb = ""
    wandb_key = os.getenv("WANDB_KEY", "")
    if wandb_key and not is_debug:
        wandb = (
            "--use-wandb "
            f"--wandb-project {os.getenv('WANDB_PROJECT', 'miles-agent-v2')} "
            f"--wandb-group {os.getenv('WANDB_GROUP', 'agent-v2')} "
            f"--wandb-key {wandb_key} "
        )
        wandb_team = os.getenv("WANDB_TEAM", "")
        if wandb_team:
            wandb += f"--wandb-team {wandb_team} "

    sglang = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 " "--use-miles-router "

    misc = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    custom = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
    )

    return ckpt + perf + rollout + grpo + optimizer + wandb + sglang + misc + custom


def main():
    parser = argparse.ArgumentParser(
        description="Agent V2: Miles <-> Harbor training launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "debug"],
        default="train",
        help="Training mode: 'train' for full run, 'debug' for minimal pipeline verification",
    )
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")

    parser.add_argument("--hf-checkpoint", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--model-script", default="qwen3-4B-Instruct-2507.sh", help="Model arch script under miles/scripts/models/"
    )
    parser.add_argument("--ref-load", default="/root/qwen3-4B-Instruct-2507_torch_dist")
    parser.add_argument("--load-dir", default="", help="Resume from checkpoint")
    parser.add_argument("--save-dir", default="/root/qwen3-4B-Instruct-2507_miles_v2/")
    parser.add_argument("--save-interval", type=int, default=100)

    parser.add_argument("--prompt-data", default="/root/swe_train.jsonl")
    parser.add_argument("--num-rollout", type=int, default=3000)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--n-samples-per-prompt", type=int, default=8)
    parser.add_argument("--rollout-temperature", type=float, default=0.8)
    parser.add_argument("--rollout-max-response-len", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=2048)

    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-loss-coef", type=float, default=0.01)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--eps-clip-high", type=float, default=0.28)

    parser.add_argument(
        "--agent-server-url",
        default=None,
        help="Harbor server URL (default: $AGENT_SERVER_URL or http://swe_env:11000)",
    )
    parser.add_argument(
        "--harbor-tasks-dir",
        default=None,
        help="Harbor tasks directory (default: $HARBOR_TASKS_DIR or /root/harbor_tasks)",
    )
    parser.add_argument(
        "--router-external-host", default=None, help="Hostname for Docker containers to reach Miles Router"
    )

    args = parser.parse_args()

    agent_server_url = args.agent_server_url or os.getenv(
        "AGENT_SERVER_URL", os.getenv("SWE_AGENT_URL", "http://swe_env:11000")
    )
    harbor_tasks_dir = args.harbor_tasks_dir or os.getenv("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    router_external_host = args.router_external_host or os.getenv("MILES_ROUTER_EXTERNAL_HOST", "")
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")

    exec_command(
        "pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 python; sleep 3; pkill -9 ray; pkill -9 python; true"
    )

    model_script = MILES_ROOT / "scripts" / "models" / args.model_script
    exec_command(f"source {model_script}")

    exec_command(
        f"ray start --head --node-ip-address {master_addr} "
        f"--num-gpus {args.num_gpus} --disable-usage-stats "
        "--dashboard-host=0.0.0.0 --dashboard-port=8265 --port=8899"
    )

    runtime_env = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": f"/root/Megatron-LM/:{SCRIPT_DIR}:{MILES_ROOT}",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
                "AGENT_SERVER_URL": agent_server_url,
                "AGENT_MODEL_NAME": os.getenv("AGENT_MODEL_NAME", "model"),
                "MILES_ROUTER_EXTERNAL_HOST": router_external_host,
                "HARBOR_TASKS_DIR": harbor_tasks_dir,
            }
        }
    )

    train_args = build_args(args)

    print(f"Launching {'debug ' if args.mode == 'debug' else ''}training...")
    print(f"  Agent server:  {agent_server_url}")
    print(f"  Harbor tasks:  {harbor_tasks_dir}")
    print(f"  Model:         {args.hf_checkpoint}")

    cmd = (
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json='{runtime_env}' "
        f"-- python3 {MILES_ROOT}/train.py "
        f"--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus} "
        f"--colocate "
        f"$(source {model_script} && echo ${{MODEL_ARGS[@]}}) "
        f"{train_args}"
    )

    result = subprocess.run(cmd, shell=True, executable="/bin/bash")
    print(f"Training {'completed' if result.returncode == 0 else 'failed'}!")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
