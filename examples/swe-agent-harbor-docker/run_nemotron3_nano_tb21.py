"""Nemotron-3-Nano-30B-A3B agentic GRPO on Harbor tasks (Terminal-Bench 2.1).

This merges two existing Miles recipes:

  * ``examples/swe-agent-harbor-docker/run.py`` supplies the agentic half --
    TITO, the Miles session server, the Harbor agent function, and the
    reward/rollout hooks.
  * ``scripts/run_nemotron_3_nano.py`` supplies the model half -- the
    ``nemotron-3-nano-30b-a3b`` Megatron model type, the AutoBridge load path,
    and the MoE rollout routing replay.

Four Nemotron-specific settings are load-bearing and differ from the
GLM-4.7-Flash launcher this is derived from:

  * ``--megatron-to-hf-mode bridge``: nemotron_h loads straight from the HF
    checkpoint through megatron.bridge, so --hf-checkpoint and --ref-load are
    the same directory and there is no torch_dist conversion step.
  * ``--use-rollout-routing-replay``: replays the rollout's expert routing in
    the training forward. Without it the sigmoid-routed MoE drifts ~0.28 in
    train-vs-rollout logprob instead of ~0.014.
  * ``--attention-backend auto``: the Mamba layers pick their own kernel.
  * ``--tito-model nemotron3`` with the nemotron_3 reasoning parser and the
    qwen3_coder tool-call parser, per Nemotron3TITOTokenizer.

Note on ``--max-tokens-per-gpu``: dynamic batching packs samples first-fit, so
a sample longer than the budget still gets a micro-batch of its own. The knob
bounds packing of short samples, not the peak activation of a long trajectory
-- only context parallelism (--context-parallel-size) or a smaller
--max-seq-len does that.

Usage:
    python run_nemotron3_nano_tb21.py --mode debug_rollout_only --num-rollout 1
    python run_nemotron3_nano_tb21.py --num-rollout 200
"""

import os
import socket
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
    megatron_model_type: str = "nemotron-3-nano-30b-a3b"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths. Bridge mode loads the HF checkpoint directly, so ref_load == hf_checkpoint.
    model_dir: str = "/scratch/260908-e3d52c5e/model"
    save_dir: str = "/scratch/260908-e3d52c5e/checkpoints"
    prompt_data: str = "/scratch/260908-e3d52c5e/tb21_train.jsonl"
    save_traces_dir: str = "/scratch/260908-e3d52c5e/traces"

    # Training settings
    max_seq_len: int = 65536
    rollout_max_response_len: int = 8192
    num_rollout: int = 200
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 20
    lr: float = 1e-6

    # Parallelism. Verified cells for this model are TP2/PP2/EP2 (default),
    # EP4, TP2xEP4, PP2xEP4 and CP2xEP4 -- all on one node of 8 GPUs.
    tp: int = 2
    pp: int = 2
    cp: int = 1
    ep: int = 2
    etp: int = 1
    max_tokens_per_gpu: int = 16384
    log_probs_chunk_size: int = 128
    optimizer_cpu_offload: bool = False

    # Agent settings
    agent_server_url: str = os.environ.get("AGENT_SERVER_URL", "http://100.98.192.125:8110")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    # Address the Harbor sandboxes dial back on; must route from the agent server.
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())
    # Bind address for the trainer's own services; the pod IP, never the tailnet name.
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")
    # The session server binds this separately: on a devbox it must be 0.0.0.0 or
    # tailnet-delivered connections are refused.
    session_server_ip: str = "0.0.0.0"
    agent_trial_timeout: int = 10800

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "nemotron3-nano-agentic")
    wandb_team: str = os.environ.get("WANDB_TEAM", "radixarkai")
    wandb_run_name: str = "260908-e3d52c5e-nemotron3-nano-moe-tb21"

    use_prometheus: bool = True
    prometheus_port: int = 9090


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir} "
        f"--ref-load {args.model_dir} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
        "--megatron-to-hf-mode bridge "
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

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pp} "
        f"--context-parallel-size {args.cp} "
        f"--expert-model-parallel-size {args.ep} "
        f"--expert-tensor-parallel-size {args.etp} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--log-probs-chunk-size {args.log_probs_chunk_size} "
    )
    if args.optimizer_cpu_offload:
        perf_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser qwen3_coder "
        "--sglang-reasoning-parser nemotron_3 "
        "--sglang-router-port 31000 "
        # Keeps train logprobs aligned with rollout logprobs for the sigmoid-routed MoE.
        "--use-rollout-routing-replay "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model nemotron3 "
        "--use-session-server "
        "--session-server-port 30000 "
        "--session-server-workers 32 "
        f"--session-server-ip {args.session_server_ip} "
    )

    observability_args = (
        f"--dump-details {args.save_traces_dir} "
        "--use-miles-dashboard "
        "--observe-training-entropy "
        "--use-rollout-entropy "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # nemotron_h is a Mamba/attention hybrid; the Mamba layers select their own kernel.
        "--attention-backend auto "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_nodes * args.num_gpus_per_node} "
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
            f"--prometheus-run-name {args.wandb_run_name} "
        )

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{observability_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars = {
        # SCRIPT_DIR is the harbor example dir, where swe_agent_function and generate live.
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        # Must stay above the agent server's own --agent-timeout, or the client
        # gives up first and the sandbox leaks a --max-concurrent slot.
        "AGENT_TRIAL_TIMEOUT": str(args.agent_trial_timeout),
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
    execute(args)


if __name__ == "__main__":
    typer.run(main)
