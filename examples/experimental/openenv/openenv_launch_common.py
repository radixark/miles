"""Shared launch helpers for the OpenEnv tbench2 learning launchers.

``run-openenv-tbench2.py`` (GLM-4.7-Flash) is the launcher in this example;
sibling per-model launchers (e.g. a DeepSeek-V4-Flash variant) reuse the same
agentic adapter and differ only in the model-family serving/training profile.
The model-agnostic fragments (process cleanup, GRPO/optimizer/rollout/agent
flags, W&B + Prometheus wiring, and the OpenEnv env-var plumbing) live here so
those launchers cannot silently drift apart. Each launcher keeps only its own
perf/sglang/misc profile and its ``ScriptArgs`` defaults.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Protocol


class LaunchArgs(Protocol):
    """The config fields the shared helpers read (satisfied by each launcher's ScriptArgs)."""

    prompt_data: str
    rollout_batch_size: int
    n_samples_per_prompt: int
    max_seq_len: int
    global_batch_size: int

    openenv_env_url: str
    agent_model_name: str
    openenv_max_turns: int
    openenv_max_rollout_time_seconds: int
    openenv_tb2_tasks_dir: str
    daytona_api_key: str
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
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
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
        "--num-rollout 40 "
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
        "--custom-agent-function-path openenv_agent_function.run "
        "--custom-rm-path openenv_generate.reward_func "
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
    return {
        "PYTHONPATH": f"{megatron_path}:{script_dir}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(args.openenv_max_rollout_time_seconds),
        "AGENT_MODEL_NAME": args.agent_model_name,
    }


def apply_optional_env_vars(env: dict[str, str], args: LaunchArgs) -> None:
    """Add host-rewrite / per-task-Daytona env vars when the args request them."""
    if args.miles_host_ip:
        env["MILES_HOST_IP"] = args.miles_host_ip
    if args.router_external_host:
        env["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host
    if args.openenv_tb2_tasks_dir:
        if not args.daytona_api_key:
            raise ValueError("DAYTONA_API_KEY required in per-task Daytona mode")
        # Preflight the lazily-imported SDK. Without this, a missing install only
        # surfaces inside each episode's sandbox start, where the failed sample is
        # aborted, the group dropped, and the rollout loop refills forever — a
        # silent GPU-burning churn instead of a launch-time error.
        try:
            import daytona  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "per-task Daytona mode needs the daytona SDK in the rollout "
                "process's environment: pip install daytona "
                "(or pip install -e '<OpenEnv>/envs/tbench2_env[daytona]')"
            ) from e
        # Same preflight for the env package the recipe bakes into each task
        # image. The import check catches a missing install; the source probe
        # catches an install that imports fine but lacks the server features
        # the per-task leg scores through (canonical tests/test.sh evaluate,
        # TB2_WITHHOLD_TESTS) — that one would not even fail per-episode, it
        # would silently mis-score every episode.
        try:
            import tbench2_env
        except ImportError as e:
            raise RuntimeError(
                "per-task Daytona mode needs tbench2_env in the rollout "
                "process's environment: pip install -e '<OpenEnv>/envs/tbench2_env' "
                "from the pinned checkout in this directory's README"
            ) from e
        server_src = Path(tbench2_env.__file__).resolve().parent / "server" / "tbench2_env_environment.py"
        src_text = server_src.read_text(encoding="utf-8") if server_src.is_file() else ""
        if "TB2_WITHHOLD_TESTS" not in src_text:
            raise RuntimeError(
                "the installed tbench2_env server lacks the per-task sandbox "
                "contract (canonical test.sh scoring / TB2_WITHHOLD_TESTS): "
                "install the pinned checkout from this directory's README, "
                "not upstream main"
            )
        env["OPENENV_TB2_TASKS_DIR"] = args.openenv_tb2_tasks_dir
        env["DAYTONA_API_KEY"] = args.daytona_api_key
