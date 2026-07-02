"""OpenEnv Terminal-Bench-2 (tbench2) learning launcher (GLM-4.7-Flash).

Same shape as ``run-openenv-coding.py`` but drives the OpenEnv tbench2 env via the
generic ``openenv_agent_function.run`` with ``OPENENV_ENV_TYPE=tbench2``. Unlike
the single-step coding/echo envs, tbench2 is *multi-turn*: the adapter runs an
agentic loop (reset(task_id) -> {policy emits a shell command -> step(exec) ->
feed output back} -> evaluate) and the reward is the binary pytest result
(1.0 all tests pass, else 0.0).

Prereqs:
    # 1. Install the env client where the rollout runs (pulls camel-ai; isolate
    #    from the training env if its deps clash with the miles image).
    pip install -e <OpenEnv>/envs/tbench2_env
    # 2. Get the TB2 task suite + build prompt-data (task_ids).
    git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git /workspace/terminal-bench-2
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 --output /root/tbench2_train.jsonl
    # 3. Serve the env. The tbench2 server supports concurrency natively via
    #    MAX_CONCURRENT_ENVS (no wrapper needed). Choose execution mode:
    #      TB2_MODE=docker  -> real TB2 fidelity (needs docker.sock + image pulls)
    #      TB2_MODE=local   -> runs in-process, ignores task Dockerfiles (degraded)
    TB2_MODE=docker TB2_TASKS_DIR=/workspace/terminal-bench-2 MAX_CONCURRENT_ENVS=32 \
        python -m tbench2_env.server.app --port 8003

    NOTE (open decisions before a real run): docker mode wants a Docker host with
    disk + socket; colocating heavy per-task containers on the GPU pod is risky,
    so the env server likely runs off-pod (use --openenv-env-url / the host
    rewrite). The binary sparse reward also needs a task subset where the base
    policy *sometimes* succeeds (advantage variance) -- e.g. the TB2 variance
    band -- or GRPO sees a flat signal.

Usage:
    python run-openenv-tbench2.py --openenv-env-url http://<env-host>:8003
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent


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
    save_dir: str = "/workspace/GLM-4.7-Flash_openenv_tbench2/"
    prompt_data: str = "/root/tbench2_train.jsonl"

    # Training settings (small; multi-turn so responses run long)
    max_seq_len: int = 16384
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32

    # OpenEnv settings
    openenv_env_url: str = os.environ.get("OPENENV_ENV_URL", "http://localhost:8003")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "30"))
    # When set, the adapter ignores --openenv-env-url and instead provisions a
    # pool of Daytona sandboxes from this snapshot, rotating episodes across them.
    openenv_daytona_snapshot: str = os.environ.get("OPENENV_DAYTONA_SNAPSHOT", "")
    openenv_daytona_pool_size: int = int(os.environ.get("OPENENV_DAYTONA_POOL_SIZE", "8"))
    openenv_daytona_port: int = int(os.environ.get("OPENENV_DAYTONA_PORT", "8000"))
    daytona_api_key: str = os.environ.get("DAYTONA_API_KEY", "")
    # When set, miles dumps full per-episode agent trajectories (tokens, logprobs,
    # loss masks, reward, multi-turn messages) to <dir>/rollout_data/{rollout_id}.pt
    # for post-hoc inspection via miles.utils.debug_utils.display_debug_rollout_data.
    dump_details: str = os.environ.get("OPENENV_DUMP_DETAILS", "")
    # Optional host rewrite for the policy URL (only needed if the in-process
    # agent cannot reach the session server at its raw base_url host).
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    # Leave empty so miles resolves the numeric LAN IP itself. sgl-router's Rust
    # binder rejects a hostname ("invalid socket address syntax"), and a numeric
    # base_url host keeps the in-process policy client off hostname DNS too.
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "openenv-tbench2-learn")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "openenv-tbench2-learn"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "openenv-tbench2-learn"


def cleanup():
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


def prepare(args: ScriptArgs):
    """Convert HF checkpoint to torch_dist format if not already done."""
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
        "--save-interval 100 "
    )

    rollout_args = (
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

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 2 "  # single 8-GPU node: TP=4 -> DP=2, so EP<=2
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
        "--custom-agent-function-path openenv_agent_function.run "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30000 "
        "--tito-allowed-append-roles user tool "
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

    dump_args = f"--dump-details {args.dump_details} " if args.dump_details else ""

    wandb_args = ""
    if args.wandb_key:
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {args.wandb_project} "
            f"--wandb-group {args.wandb_run_name} "
            f"--wandb-key {args.wandb_key} "
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
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
        f"{dump_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENENV_ENV_TYPE": "tbench2",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "AGENT_MODEL_NAME": args.agent_model_name,
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip
    if args.router_external_host:
        extra_env_vars["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host
    if args.openenv_daytona_snapshot:
        assert args.daytona_api_key, "DAYTONA_API_KEY required when openenv_daytona_snapshot is set"
        extra_env_vars["OPENENV_DAYTONA_SNAPSHOT"] = args.openenv_daytona_snapshot
        extra_env_vars["OPENENV_DAYTONA_POOL_SIZE"] = str(args.openenv_daytona_pool_size)
        extra_env_vars["OPENENV_DAYTONA_PORT"] = str(args.openenv_daytona_port)
        extra_env_vars["DAYTONA_API_KEY"] = args.daytona_api_key

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
