"""Train GLM-4.7-Flash with Terminus 2 trajectory compaction.

This one-node recipe uses Harbor for Terminal-Bench 2 environments and Miles'
session server v2 to retain the trajectory segments created by Terminus 2
summarization. Start the Harbor agent server as described in the adjacent
README before launching this script.

The default recipe runs 100 GRPO steps over all 89 Terminal-Bench 2 tasks, with
four prompts and eight independent rollouts per prompt in each step.

Example:
    python examples/experimental/terminus-compaction/run.py \
        --skip-prepare \
        --model-dir /path/to/models \
        --output-dir /path/to/output \
        --agent-server-url http://agent-server.example:11000 \
        --session-server-ip 0.0.0.0 \
        --router-external-host trainer.example
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_HARBOR_PIPELINE_DIR = U.repo_base_dir / "examples" / "swe-agent-harbor-docker"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = field(default_factory=U.create_run_id)
    megatron_model_type: str = "glm4.7-flash"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    skip_prepare: bool = False
    model_dir: str = "/root/models"
    model_name: str = "GLM-4.7-Flash"
    hf_checkpoint: str = ""
    ref_load: str = ""
    save_dir: str = ""
    save_traces_dir: str = ""
    prompt_data: str = "/root/tb2_train_89.jsonl"

    max_seq_len: int = 32768
    rollout_max_response_len: int = 8192
    num_rollout: int = 100
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 100

    agent_server_url: str = field(default_factory=lambda: os.environ.get("AGENT_SERVER_URL", "http://127.0.0.1:11000"))
    agent_model_name: str = field(default_factory=lambda: os.environ.get("AGENT_MODEL_NAME", "model"))
    agent_trial_timeout: int = 7200
    router_external_host: str = field(default_factory=lambda: os.environ.get("MILES_ROUTER_EXTERNAL_HOST", ""))
    miles_host_ip: str = field(default_factory=lambda: os.environ.get("MILES_HOST_IP", ""))
    session_server_ip: str = field(default_factory=lambda: os.environ.get("MILES_SESSION_SERVER_IP", ""))

    use_prometheus: bool = True
    prometheus_port: int = 9090


def _hf_checkpoint(args: ScriptArgs) -> str:
    return args.hf_checkpoint or str(Path(args.model_dir) / args.model_name)


def _ref_load(args: ScriptArgs) -> str:
    return args.ref_load or str(Path(args.model_dir) / f"{args.model_name}_torch_dist")


def _save_dir(args: ScriptArgs) -> str:
    return args.save_dir or str(Path(args.output_dir) / args.run_id / "checkpoints")


def _save_traces_dir(args: ScriptArgs) -> str:
    return args.save_traces_dir or str(Path(args.output_dir) / args.run_id / "details")


def _checkpoint_args(args: ScriptArgs) -> str:
    return (
        f"--hf-checkpoint {_hf_checkpoint(args)} "
        f"--ref-load {_ref_load(args)} "
        f"--save {_save_dir(args)} "
        f"--save-interval {args.save_interval} "
    )


def _rollout_args(args: ScriptArgs) -> str:
    return (
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


def _performance_args(args: ScriptArgs) -> str:
    world_size = args.num_nodes * args.num_gpus_per_node
    return (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {world_size} "
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


def _grpo_args() -> str:
    return (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )


def _optimizer_args() -> str:
    return (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )


def _sglang_args() -> str:
    return (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-router-port 31000 "
    )


def _agent_args(args: ScriptArgs) -> str:
    bind_arg = f"--session-server-ip {args.session_server_ip} " if args.session_server_ip else ""
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server v2 "
        f"{bind_arg}"
        "--session-server-port 30000 "
        "--session-server-workers 32 "
    )


def _misc_args(args: ScriptArgs) -> str:
    debug_arg = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    return (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--observe-training-entropy "
        "--log-multi-turn "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"{debug_arg}"
    )


def _observability_args(args: ScriptArgs) -> str:
    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = (
            "--use-prometheus " f"--prometheus-port {args.prometheus_port} " f"--prometheus-run-name {args.run_id} "
        )
    return (
        f"--dump-details {_save_traces_dir(args)} "
        "--use-miles-dashboard "
        "--use-rollout-entropy "
        f"{prometheus_args}"
        f"{U.get_default_wandb_args(__file__, run_name_prefix='terminus-compaction', run_id=args.run_id)}"
    )


def _build_train_args(args: ScriptArgs) -> str:
    return "".join(
        (
            _checkpoint_args(args),
            _rollout_args(args),
            _optimizer_args(),
            _grpo_args(),
            _observability_args(args),
            _performance_args(args),
            _sglang_args(),
            _agent_args(args),
            _misc_args(args),
        )
    )


def _extra_env_vars(args: ScriptArgs) -> dict[str, str]:
    env = {
        "PYTHONPATH": str(_HARBOR_PIPELINE_DIR),
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "AGENT_TRIAL_TIMEOUT": str(args.agent_trial_timeout),
    }
    if args.router_external_host:
        env["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host
    if args.miles_host_ip:
        env["MILES_HOST_IP"] = args.miles_host_ip
    return env


def _prepare(args: ScriptArgs) -> None:
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=_hf_checkpoint(args),
        megatron_path=args.megatron_path,
    )


def _execute(args: ScriptArgs) -> None:
    U.execute_train(
        train_args=_build_train_args(args),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=_extra_env_vars(args),
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    if not args.skip_prepare:
        _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
