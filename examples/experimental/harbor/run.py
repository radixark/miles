"""Launcher: GLM-4.7-Flash GRPO on Harbor tasks, with Harbor run in-process on a
cloud sandbox backend (no agent server).

Usage:
    HARBOR_ENV_TYPE=e2b python run.py --harbor-tasks-dir /path/to/harbor_tasks ...

The trainer side is examples/swe-agent-harbor-docker/run.py with the agent
server swapped for harbor_agent_function.run; the reward hook and metrics come
from that example (generate.py), which this launcher puts on PYTHONPATH.
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from launch_common import harbor_env_vars

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent
# reward_func / RolloutFn (agent-metric aggregation) are shared with the agent-server example
HARBOR_DOCKER_EXAMPLE_DIR = SCRIPT_DIR.parents[1] / "swe-agent-harbor-docker"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "GLM-4.7-Flash"
    hf_checkpoint: str = "zai-org/GLM-4.7-Flash"
    ref_load: str = "/root/GLM-4.7-Flash_torch_dist"
    save_dir: str = "/root/GLM-4.7-Flash_harbor/"
    prompt_data: str = "/root/tb2_train.jsonl"

    # Training settings
    max_seq_len: int = 65536
    num_rollout: int = 3000
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 100
    save_traces_dir: str = ""

    # Harbor settings (see harbor_agent_function.py for what each does)
    harbor_env_type: str = os.environ.get("HARBOR_ENV_TYPE", "")
    harbor_env_kwargs: str = os.environ.get("HARBOR_ENV_KWARGS", "")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    harbor_trials_dir: str = os.environ.get("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    agent_timeout: int = int(os.environ.get("AGENT_TIMEOUT", "5400"))
    # provider key files; the launcher forwards the PATH, workers read the file
    daytona_api_key_file: str = os.environ.get("DAYTONA_API_KEY_FILE", "")
    e2b_api_key_file: str = os.environ.get("E2B_API_KEY_FILE", "")
    modal_config_file: str = os.environ.get("MODAL_CONFIG_PATH", "")

    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "my-wandb-project")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm47-flash-harbor-inprocess"


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run(f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true", shell=True)
    time.sleep(5)


def prepare(args: ScriptArgs):
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
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
        "--rollout-max-response-len 8192 "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
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
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-router-port 31000 "
    )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path harbor_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30000 "
        "--session-server-workers 32 "
    )
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_gpus_per_node} "
    )
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    trace_args = f"--dump-details {args.save_traces_dir} " if args.save_traces_dir else ""
    wandb_args = ""
    if args.wandb_key:
        wandb_args = f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.wandb_run_name} --wandb-key {args.wandb_key} "
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    train_args = (
        f"{ckpt_args}{rollout_args}{optimizer_args}{grpo_args}{wandb_args}{trace_args}"
        f"{perf_args}{sglang_args}{agent_args}{misc_args}{debug_args}"
    )

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{HARBOR_DOCKER_EXAMPLE_DIR}:{U.repo_base_dir}",
        **harbor_env_vars(args),
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
