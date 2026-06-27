"""OpenEnv SWE-bench (Verified) learning launcher (GLM-4.7-Flash).

Same shape as ``run-openenv-tbench2.py`` but drives the OpenEnv swe env via the
generic ``openenv_agent_function.run`` with ``OPENENV_ENV_TYPE=swe``. Like
tbench2, swe is *multi-turn*: the adapter runs an agentic loop (reset(task_id) ->
{policy emits a shell command -> step(exec) -> feed output back} -> evaluate) and
the reward is the binary swebench verifier result (1.0 if the task is resolved,
else 0.0).

Prereqs:
    # 1. Make the swe_env client importable where the rollout runs. The package
    #    lives next to this script (examples/experimental/openenv/swe_env), and
    #    the launcher already puts that dir on PYTHONPATH, so no pip install is
    #    needed -- just ``import openenv`` (openenv[core]) must resolve.
    # 2. Build the prompt-data (task_ids) from a harbor-format SWE-bench task set.
    python make_swe_data.py --tasks_dir /home/ubuntu/harbor_tasks_swebench_verified \
        --output /root/swe_train.jsonl --n 100 --seed 0
    # 3. Serve the env (off-pod, on a Docker host with the swebench eval images):
    SWE_TASKS_DIR=/home/ubuntu/harbor_tasks_swebench_verified MAX_CONCURRENT_ENVS=16 \
        python -m swe_env.server.app --port 8004

    NOTE: the swebench eval images are heavy (~1-2 GB each) and each episode runs
    its own container, so the env server runs off the GPU pod; the policy reaches
    it via --openenv-env-url (e.g. a Tailscale egress host). The binary sparse
    reward also needs a task subset where the base policy *sometimes* succeeds
    (advantage variance) or GRPO sees a flat signal.

Usage:
    python run-openenv-swe.py --openenv-env-url http://<env-host>:8004
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
    save_dir: str = "/workspace/GLM-4.7-Flash_openenv_swe/"
    prompt_data: str = "/root/swe_train.jsonl"

    # Training settings (small; multi-turn so responses run long)
    max_seq_len: int = 16384
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32

    # OpenEnv settings
    openenv_env_url: str = os.environ.get("OPENENV_ENV_URL", "http://localhost:8004")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    # Optional path to an OpenEnv source checkout (its ``src`` dir) to prepend to
    # PYTHONPATH. Use when ``openenv`` is not pip-installed in the rollout env --
    # e.g. on an ephemeral pod where the install does not survive restarts; point
    # this at a checkout on a persistent volume instead.
    openenv_src_path: str = os.environ.get("OPENENV_SRC_PATH", "")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "30"))
    # Optional host rewrite for the policy URL (only needed if the in-process
    # agent cannot reach the session server at its raw base_url host).
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    # Leave empty so miles resolves the numeric LAN IP itself. sgl-router's Rust
    # binder rejects a hostname ("invalid socket address syntax"), and a numeric
    # base_url host keeps the in-process policy client off hostname DNS too.
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "openenv-swe-learn")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "openenv-swe-learn"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "openenv-swe-learn"


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
        "--max-tokens-per-gpu 8192 "
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
    )

    miles_root = U.repo_base_dir

    pythonpath = f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}"
    if args.openenv_src_path:
        pythonpath = f"{args.openenv_src_path}:{pythonpath}"

    extra_env_vars = {
        "PYTHONPATH": pythonpath,
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENENV_ENV_TYPE": "swe",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "AGENT_MODEL_NAME": args.agent_model_name,
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip
    if args.router_external_host:
        extra_env_vars["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host

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
