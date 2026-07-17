"""OpenEnv SWE-bench-style learning launcher (GLM-4.7-Flash).

Drives the OpenEnv env via ``swebench_agent_function.run`` over the SWE-Rebench-V2
"donor" task family. This is *multi-turn*: the adapter runs an agentic loop
(reset(task_id) -> {policy emits a shell command -> step(exec) -> feed output
back} -> verify) and the reward is the binary verifier result (1.0 if the task's
test_command passes, else 0.0).

Prereqs:
    # 1. Install the env client where the rollout runs (pulls camel-ai; isolate
    #    from the training env if its deps clash with the miles image). This is
    #    the same generic OpenEnv tbench2 env server the TB2 example uses.
    pip install -e <OpenEnv>/envs/tbench2_env
    # 2. Make each task pullable. The env server only PULLS the image named in
    #    task.toml [environment].docker_image; SWE-Rebench-V2 donor tasks ship an
    #    environment/Dockerfile instead. Build each task's image, push it to a
    #    registry the env host can reach, and set docker_image in its task.toml.
    #    (Offline data prep -- no code change; see README for the loop.)
    # 3. Build prompt-data (task_ids) from the pool.
    python make_swebench_data.py --tasks_dir /root/swebench_pool --output /root/swebench_train.jsonl
    # 4. Serve the env pointing at the pool dir.
    TB2_MODE=docker TB2_TASKS_DIR=/root/swebench_pool MAX_CONCURRENT_ENVS=32 \
        python -m tbench2_env.server.app --port 8003

    NOTE: the binary sparse reward needs a task subset where the base policy
    *sometimes* succeeds (advantage variance), or GRPO sees a flat signal.

Usage:
    python run-openenv-swebench.py --openenv-env-url http://<env-host>:8003
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import swebench_launch_common as C
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
    save_dir: str = "/workspace/GLM-4.7-Flash_openenv_swebench/"
    prompt_data: str = "/root/swebench_train.jsonl"

    # Training settings (small; multi-turn so responses run long)
    max_seq_len: int = 16384
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32

    # OpenEnv settings
    openenv_env_url: str = os.environ.get("OPENENV_ENV_URL", "http://localhost:8003")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "30"))
    # Hard wall-clock cap (seconds) per episode. An episode that does not return
    # within the limit is terminated and scored reward 0, bounding long-trajectory
    # stragglers that would otherwise stall the whole rollout batch.
    openenv_max_rollout_time_seconds: int = int(os.environ.get("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "3600"))
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
    wandb_project: str = os.environ.get("WANDB_PROJECT", "openenv-swebench-learn")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "openenv-swebench-learn"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "openenv-swebench-learn"


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

    rollout_args = C.rollout_args(args)

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

    grpo_args = C.grpo_args()

    optimizer_args = C.optimizer_args()

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-router-port 31000 "
    )

    agent_args = C.agent_args("glm47")

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

    wandb_args = C.wandb_args(args)

    prometheus_args = C.prometheus_args(args)

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

    extra_env_vars = C.base_env_vars(args, str(SCRIPT_DIR), args.megatron_path, U.repo_base_dir)
    C.apply_optional_env_vars(extra_env_vars, args)

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
    C.cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
