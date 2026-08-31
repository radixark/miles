"""Qwen3-Coder SWE-bench agentic training launcher (AMD ROCm / MI350X).

Standalone launcher for Qwen3-Coder-30B-A3B on Harbor, supporting both
colocate and async training. Tested on a single node.

Pass --async-mode to enable async training: on one 8-GPU node it runs a 4+4
split -- 4 GPUs run Megatron training and the other 4 run SGLang rollout
concurrently (train_async.py). Without --async-mode, colocate mode shares all
GPUs between training and rollout in the same process (train.py).

Pass --mode debug_rollout_only to run rollout generation only, with no
Megatron. The default, --mode normal, runs the full train + rollout loop.

Prerequisites:
    - Prompt dataset (.jsonl) prepared and available at --prompt-data
    - HF checkpoint converted to torch_dist (run without --skip-prepare first)
    - harbor-server container running on swe-net
    - /data/miles_ci/trials writable

Every invocation must supply the compulsory arguments enforced by
ScriptArgs.__post_init__ (via CLI flag or the matching env var), else it
raises "Required fields not set":
    --megatron-path (MEGATRON_PATH), --hf-checkpoint (HF_CHECKPOINT),
    --ref-load (MILES_REF_LOAD), --save-dir (MILES_SAVE_DIR),
    --prompt-data (MILES_PROMPT_DATA), --harbor-tasks-dir (HARBOR_TASKS_DIR)

Usage (add the compulsory arguments above to each):
    # Colocate (default): all GPUs shared between training and rollout
    python run-qwen3-swe.py --skip-prepare \\
        --prompt-data /root/datasets/swe_gym_lite_clean.jsonl

    # Async: --async-mode enables the 4+4 split (4 train + 4 rollout) on one node
    python run-qwen3-swe.py --async-mode --skip-prepare \\
        --prompt-data /root/datasets/swe_gym_lite_clean.jsonl \\
        --wandb-project qwen3-coder-swe \\
        --agent-server-url http://harbor-server:30000 \\
        --router-external-host miles-trainer \\
        --miles-trials-dir /data/miles_ci/trials \\
        --num-rollout 15 --save-interval 15
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent

# True when running on AMD ROCm (HIP); False on NVIDIA CUDA.
IS_ROCM: bool = getattr(torch.version, "hip", None) is not None


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    num_gpus_per_node: int = 8
    megatron_path: str = os.environ.get("MEGATRON_PATH", "")

    # Async / disaggregated mode: split GPUs between training and rollout.
    # train_num_gpus GPUs run Megatron; the rest run SGLang concurrently.
    async_mode: bool = False
    train_num_gpus: int = 4

    skip_prepare: bool = False
    hf_checkpoint: str = os.environ.get("HF_CHECKPOINT", "")
    ref_load: str = os.environ.get("MILES_REF_LOAD", "")
    save_dir: str = os.environ.get("MILES_SAVE_DIR", "")
    prompt_data: str = os.environ.get("MILES_PROMPT_DATA", "")

    # Training settings
    max_seq_len: int = 32768
    num_rollout: int = 3000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    save_interval: int = 100
    save_traces_dir: str = os.environ.get("MILES_SAVE_TRACES_DIR", "")

    # Agent settings
    agent_server_url: str = os.environ.get(
        "AGENT_SERVER_URL", os.environ.get("SWE_AGENT_URL", "http://harbor-server:30000")
    )
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "")
    miles_trials_dir: str = os.environ.get("MILES_TRIALS_DIR", "")

    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "qwen3-coder-swe")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "qwen3-coder-30b-swe-gym-lite"
    # offline: writes locally, never blocks on network (prevents wandb.log() deadlock
    # in multi-rank shared mode); sync manually with `wandb sync --legacy <run-dir>`
    wandb_mode: str = os.environ.get("WANDB_MODE", "offline")
    wandb_dir: str = os.environ.get("WANDB_DIR", "")

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "qwen3-coder-30b-swe"

    # Extra env vars passed through to the ray job (e.g. SGLANG_USE_AITER)
    extra_env_vars: str = ""

    def __post_init__(self):
        required = {
            "megatron_path": self.megatron_path,
            "hf_checkpoint": self.hf_checkpoint,
            "ref_load": self.ref_load,
            "save_dir": self.save_dir,
            "prompt_data": self.prompt_data,
            "harbor_tasks_dir": self.harbor_tasks_dir,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Required fields not set: {', '.join(missing)}")
        if self.async_mode and self.rollout_gpus <= 0:
            raise ValueError(
                f"--train-num-gpus {self.train_num_gpus} >= --num-gpus-per-node "
                f"{self.num_gpus_per_node}: no GPUs left for rollout"
            )
        # Megatron training doesn't launch in debug_rollout_only mode, so its
        # tensor/expert-parallel divisibility (and train_gpus >= 1) don't apply.
        if self.mode != "debug_rollout_only":
            if self.train_gpus < 1:
                raise ValueError(f"--train-num-gpus must be >= 1, got {self.train_num_gpus}")
            if self.train_gpus % self.tp_size != 0:
                raise ValueError(
                    f"train GPUs {self.train_gpus} not divisible by tp_size {self.tp_size}: "
                    f"invalid Megatron tensor parallelism config"
                )
            if self.train_gpus % self.ep_size != 0:
                raise ValueError(
                    f"train GPUs {self.train_gpus} not divisible by ep_size {self.ep_size}: "
                    f"invalid Megatron expert parallelism config"
                )

    # Single-node topology: async = train_num_gpus for Megatron and the rest for
    # SGLang (concurrent); colocate = all GPUs shared between training and rollout.
    @property
    def train_gpus(self) -> int:
        return self.train_num_gpus if self.async_mode else self.num_gpus_per_node

    @property
    def rollout_gpus(self) -> int:
        if self.async_mode:
            return self.num_gpus_per_node - self.train_num_gpus
        return self.num_gpus_per_node

    # TP=2, EP=4 for async; TP=1, EP=8 for colocate. Both clamped to the GPU
    # count so a sub-8-GPU node stays valid (TP/EP must divide the world size).
    @property
    def tp_size(self) -> int:
        return min(2 if self.async_mode else 1, self.train_gpus)

    @property
    def ep_size(self) -> int:
        return min(4 if self.async_mode else 8, self.train_gpus)


def _model_cfg() -> tuple[str, str]:
    """Return (megatron_model_type, model_name)."""
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "10000000"
    return "qwen3-30B-A3B", "Qwen3-Coder-30B-A3B-Instruct"


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "train_async.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        pattern = f"[{t[0]}]{t[1:]}"
        subprocess.run(
            f"pgrep -f '{pattern}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def prepare(args: ScriptArgs):
    """Convert HF checkpoint to torch_dist format if not already done."""
    megatron_model_type, model_name = _model_cfg()
    U.convert_checkpoint(
        model_name=model_name,
        megatron_model_type=megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=str(Path(args.ref_load).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    megatron_model_type, model_name = _model_cfg()

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
        f"--tensor-model-parallel-size {args.tp_size} "
        + ("--sequence-parallel " if args.tp_size > 1 else "")
        + "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.ep_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
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
        f"--rollout-num-gpus-per-engine {args.rollout_gpus} "
        + (
            "--sglang-mem-fraction-static 0.5 "
            if (IS_ROCM and not args.async_mode)
            else "--sglang-mem-fraction-static 0.7 "
        )
        + "--sglang-cuda-graph-max-bs 512 "
        + ("--sglang-moe-runner-backend triton " if IS_ROCM else "")
        + f"--sglang-context-length {args.max_seq_len} "
        "--sglang-tool-call-parser qwen25 "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-router-port 31000 "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model qwen3 "
        "--use-session-server "
        "--session-server-port 30000 "
    )

    if args.async_mode:
        placement_args = (
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node {args.train_gpus} "
            f"--rollout-num-gpus {args.rollout_gpus * args.num_nodes} "
        )
    else:
        offload_flags = "--no-offload-train --no-offload-rollout " if IS_ROCM else ""
        placement_args = (
            "--colocate "
            f"{offload_flags}"
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"{placement_args}"
    )

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    trace_args = f"--dump-details {args.save_traces_dir} " if args.save_traces_dir else ""

    wandb_args = ""
    if args.wandb_key:
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {args.wandb_project} "
            f"--wandb-group {args.wandb_run_name} "
            f"--wandb-key {args.wandb_key} "
            f"--wandb-mode {args.wandb_mode} "
        )
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = (
            "--use-prometheus "
            f"--prometheus-port {args.prometheus_port} "
            f"--prometheus-run-name {args.prometheus_run_name} "
        )

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{trace_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars: dict[str, str] = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        "MILES_TRIALS_DIR": args.miles_trials_dir,
        "WANDB_DIR": args.wandb_dir,
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip

    train_script = "train_async.py" if args.async_mode else "train.py"

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
        train_script=train_script,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
