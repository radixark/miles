"""GLM-4.7-Flash fully-async agentic training with SWE-bench data.

Disaggregated fully-async variant for agentic tasks: training and rollout run
on separate nodes concurrently. Uses train_async.py and the fully_async_rollout
module so that weight updates do not block generation. Agent tasks are dispatched
to a Harbor-based agent server.

GLM-4.7-Flash architecture: 47 layers, 20 attention heads, 64 routed experts,
hidden_size=2048, first_k_dense_replace=1. TP must divide 20 (valid: 1,2,4,5).
Default split: 1 node training + 7 nodes inference (configurable via
--train-num-nodes), sized for an 8-node job.

Data preparation (run separately before training):
    python download_and_process_data.py \\
        --input SWE-bench/SWE-bench_Verified \\
        --output /root/swe_train.jsonl \\
        --agent-name mini-swe-agent --split test

Usage:
    python run-glm47-flash-agentic-async.py --num-nodes 8
    python run-glm47-flash-agentic-async.py --num-nodes 8 --train-num-nodes 1
    python run-glm47-flash-agentic-async.py --num-nodes 8 \\
        --agent-server-url http://ts-egress-aws-agent-server:8080
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
FULLY_ASYNC_DIR = (Path(__file__).resolve().parent.parent.parent / "fully_async").resolve()

# Cluster-wide GPU-node ceiling for the ckpt-conversion job. Kept below the
# raw node count so ckpt conversion doesn't starve the rest of the cluster.
MAX_CONVERT_GPUS = 92


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    model_name: str = "GLM-4.7-Flash"
    hf_checkpoint: str = "/models/zai-org/GLM-4.7-Flash"
    ref_load: str = "/models/zai-org/GLM-4.7-Flash_torch_dist"
    save_dir: str = "/root/GLM-4.7-Flash_agentic_async/"
    # Directory to dump rollout + training traces (per-rollout .pt files). Empty
    # means default to ``<save_dir>/traces``; set to ``"disabled"`` to skip.
    save_traces_dir: str = ""
    prompt_data: str = "/root/swe_train.jsonl"
    max_seq_len: int = 16384
    rollout_max_response_len: int = 8192
    save_interval: int = 5

    # Rollout / training batch sizing (overridable for smoke tests)
    num_rollout: int = 3000
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 4
    global_batch_size: int = 32
    over_sampling_batch_size: int = 64

    # Rollout precision
    rollout_fp8: bool = False
    rollout_health_check_first_wait: int = 1800

    # Agent settings
    agent_server_url: str = os.environ.get("AGENT_SERVER_URL", "http://ts-egress-aws-agent-server:8080")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # Disaggregated fully-async settings
    train_num_nodes: int = 1
    pause_generation_mode: Literal["in_place", "retract"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast", "p2p"] = "broadcast"
    accumulate_allreduce_grads_in_fp32: bool = False
    max_tokens_per_gpu: int = 8192
    optimizer_cpu_offload: bool = True
    use_precision_aware_optimizer: bool = True

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "glm47-flash-agentic")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm47-flash-swe-async"
    disable_wandb_random_suffix: bool = True

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "glm47-flash-swe-async"


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "train_async.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        # Bracket-wrap the first char so the pgrep pattern doesn't match its
        # own shell/subprocess command line (which literally contains the
        # bracketed pattern and thus fails the regex).
        pattern = f"[{t[0]}]{t[1:]}"
        subprocess.run(
            f"pgrep -f '{pattern}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def prepare(args: ScriptArgs):
    """Convert HF checkpoint to torch_dist format."""
    max_convert_nodes = MAX_CONVERT_GPUS // args.num_gpus_per_node
    convert_nodes = min(args.num_nodes, max_convert_nodes)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=True,
        num_nodes=convert_nodes,
        dir_dst=str(Path(args.ref_load).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    if args.pause_generation_mode == "in_place" and args.update_weight_transfer_mode == "p2p":
        raise ValueError(
            "in_place + p2p is not supported: P2P transfer engine conflicts with "
            "active NCCL inference. Use broadcast with in_place, or retract with p2p."
        )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        "--rollout-function-path fully_async_rollout.generate_rollout_fully_async "
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
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--pause-generation-mode {args.pause_generation_mode} "
    )

    eval_args = ""

    # Disaggregated split: training on train_num_nodes, inference on the rest.
    rollout_num_nodes = args.num_nodes - args.train_num_nodes
    assert rollout_num_nodes > 0, (
        f"train_num_nodes ({args.train_num_nodes}) must be less than "
        f"num_nodes ({args.num_nodes}) to leave room for inference"
    )
    train_gpus = args.train_num_nodes * args.num_gpus_per_node
    rollout_gpus = rollout_num_nodes * args.num_gpus_per_node
    print(
        f"Disagg split: {args.train_num_nodes} nodes ({train_gpus} GPUs) training, "
        f"{rollout_num_nodes} nodes ({rollout_gpus} GPUs) inference"
    )

    # Training parallelism for Flash: TP=4 (divides 20-head attention), PP=1,
    # EP = largest divisor of 64 that also divides DP.
    tp, pp = 4, 1
    dp = train_gpus // (tp * pp)
    assert train_gpus % (tp * pp) == 0, f"train GPUs ({train_gpus}) must be divisible by TP*PP ({tp * pp})"
    num_experts = 64
    ep = max(d for d in range(1, dp + 1) if num_experts % d == 0 and dp % d == 0)

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {pp} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ep} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )
    if args.optimizer_cpu_offload:
        perf_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    if args.use_precision_aware_optimizer:
        perf_args += "--use-precision-aware-optimizer "

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

    # SGLang: single-node engines with DP-attention. Flash has 20 attention
    # heads so TP=8 crashes (20 % 8 != 0); we use attn_tp=4, attn_dp=2 while
    # MoE stays 8-way TP/EP across the 8 GPUs in each rollout node.
    sglang_nodes_per_engine = 1
    sglang_world_size = sglang_nodes_per_engine * args.num_gpus_per_node
    num_engines = rollout_num_nodes // sglang_nodes_per_engine
    assert rollout_num_nodes % sglang_nodes_per_engine == 0, (
        f"rollout nodes ({rollout_num_nodes}) must be divisible by "
        f"sglang_nodes_per_engine ({sglang_nodes_per_engine})"
    )
    print(f"Inference: {num_engines} engines x {sglang_world_size} GPUs/engine")
    sglang_decode_max_bs = 256
    sglang_attn_tp_size = 4
    assert sglang_world_size % sglang_attn_tp_size == 0, (
        f"sglang world ({sglang_world_size}) must be divisible by " f"attn_tp_size ({sglang_attn_tp_size})"
    )
    sglang_attn_dp_size = sglang_world_size // sglang_attn_tp_size

    sglang_p2p_extra = ""
    if args.update_weight_transfer_mode == "p2p":
        sglang_p2p_extra = "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.80 "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-enable-dp-attention "
        f"--sglang-dp-size {sglang_attn_dp_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
        f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--use-miles-router "
        "--sglang-router-port 31000 "
        f"{sglang_p2p_extra}"
    )
    sglang_extra_env_vars: dict[str, str] = {}

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30000 "
        "--tito-allowed-append-roles user tool "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--update-weight-transfer-mode {args.update_weight_transfer_mode} "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.train_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_gpus} "
        "--grad-reduce-in-bf16 "
        "--use-fault-tolerance "
        f"--rollout-health-check-first-wait {args.rollout_health_check_first_wait} "
    )
    if args.accumulate_allreduce_grads_in_fp32:
        misc_args += "--accumulate-allreduce-grads-in-fp32 "

    traces_dir = args.save_traces_dir or f"{args.save_dir.rstrip('/')}/traces"
    if traces_dir != "disabled":
        misc_args += f"--dump-details {traces_dir} "

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
        if args.disable_wandb_random_suffix:
            wandb_args += "--disable-wandb-random-suffix "

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
        f"{eval_args}"
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

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{FULLY_ASYNC_DIR}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK", "0"),
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        **sglang_extra_env_vars,
    }
    if args.router_external_host:
        extra_env_vars["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
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
