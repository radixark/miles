"""Shared launch helpers for the OpenEnv SWE-bench-style learning launchers.

``run-openenv-swebench.py`` (GLM-4.7-Flash) is the launcher in this example;
sibling per-model launchers reuse the same agentic adapter and differ only in the
model-family serving/training profile. The model-agnostic fragments (process
cleanup, GRPO/optimizer/rollout/agent flags, W&B + Prometheus wiring, and the
OpenEnv env-var plumbing) live here so those launchers cannot silently drift
apart. Each launcher keeps only its own perf/sglang/misc profile and its
``ScriptArgs`` defaults.
"""

import os
import subprocess
import time
from typing import Protocol

# Agent-side overrides that swebench_agent_function reads straight from os.getenv
# but that base_env_vars does not derive from a launcher flag. They must be
# forwarded explicitly: with MILES_SCRIPT_EXTERNAL_RAY=1 the rollout runs on a
# remote Ray cluster that does NOT inherit the submission shell's environment, so
# an override set only in the shell would silently revert to its default on the
# workers. Forward each one that is actually set (membership, not truthiness, so
# an intentional empty value like OPENENV_CONDA_ENV="" still propagates).
_AGENT_PASSTHROUGH_ENV_VARS = (
    "OPENENV_TASK_WORKDIR",
    "OPENENV_CONDA_ENV",
    "OPENENV_MESSAGE_TIMEOUT_S",
    "OPENENV_SWEBENCH_TESTS_SRC",
    "OPENENV_SWEBENCH_TESTS_SNAPSHOT",
    "OPENENV_EVAL_CMD",
)


class LaunchArgs(Protocol):
    """The config fields the shared helpers read (satisfied by each launcher's ScriptArgs)."""

    prompt_data: str
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    max_seq_len: int
    global_batch_size: int

    openenv_env_url: str
    agent_model_name: str
    openenv_max_turns: int
    openenv_max_rollout_time_seconds: int
    router_external_host: str
    miles_host_ip: str

    wandb_key: str
    wandb_project: str
    wandb_team: str
    wandb_run_name: str

    use_prometheus: bool
    prometheus_port: int
    prometheus_run_name: str


def cleanup() -> None:
    """Kill stale training/rollout processes to free GPU resources before a run.

    ASSUMES a disposable, single-tenant pod: this example colocates training and
    rollout on all 8 GPUs of one node, so any lingering ``sglang`` / ``train.py``
    / ``MegatronTrain`` process is a leftover from a previous crashed run on this
    same box and must be reaped or it holds the GPUs. To avoid collateral damage
    on a shared host we scope the kill to the current user's own processes
    (``pgrep -u <euid>``) and still exclude this launcher and its parent. Set
    ``OPENENV_SKIP_CLEANUP=1`` to disable entirely (e.g. when another of your jobs
    is intentionally running on the same node)."""
    if os.getenv("OPENENV_SKIP_CLEANUP") == "1":
        print("Cleanup skipped (OPENENV_SKIP_CLEANUP=1)")
        return
    my_pid = os.getpid()
    ppid = os.getppid()
    euid = os.geteuid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid}, euid={euid})")
    targets = ["sglang", "train.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        subprocess.run(
            f"pgrep -u {euid} -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def rollout_args(args: LaunchArgs) -> str:
    return (
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


def grpo_args() -> str:
    return (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )


def optimizer_args() -> str:
    return (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )


def agent_args(tito_model: str) -> str:
    """Agentic-rollout wiring. Only the TITO surface differs across models."""
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swebench_agent_function.run "
        "--custom-rm-path swebench_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--tito-model {tito_model} "
        "--use-session-server "
        "--session-server-port 30000 "
        "--tito-allowed-append-roles user tool "
    )


def wandb_args(args: LaunchArgs) -> str:
    if not args.wandb_key:
        return ""
    out = (
        "--use-wandb "
        f"--wandb-project {args.wandb_project} "
        f"--wandb-group {args.wandb_run_name} "
        f"--wandb-key {args.wandb_key} "
    )
    if args.wandb_team:
        out += f"--wandb-team {args.wandb_team} "
    return out


def prometheus_args(args: LaunchArgs) -> str:
    if not args.use_prometheus:
        return ""
    return (
        "--use-prometheus "
        f"--prometheus-port {args.prometheus_port} "
        f"--prometheus-run-name {args.prometheus_run_name} "
    )


def base_env_vars(args: LaunchArgs, script_dir: str, megatron_path: str, miles_root: str) -> dict[str, str]:
    env = {
        "PYTHONPATH": f"{megatron_path}:{script_dir}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(args.openenv_max_rollout_time_seconds),
        "AGENT_MODEL_NAME": args.agent_model_name,
    }
    # Forward advertised agent-side overrides so they survive to a remote Ray
    # cluster (MILES_SCRIPT_EXTERNAL_RAY=1); see _AGENT_PASSTHROUGH_ENV_VARS.
    for name in _AGENT_PASSTHROUGH_ENV_VARS:
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def apply_optional_env_vars(env: dict[str, str], args: LaunchArgs) -> None:
    """Add host-rewrite env vars when the args request them."""
    if args.miles_host_ip:
        env["MILES_HOST_IP"] = args.miles_host_ip
    if args.router_external_host:
        env["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host
