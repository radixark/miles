"""Qwen3.5-4B fully-async code-agent training with SWE-bench data.

Disaggregated fully-async variant for code-agent tasks: training and rollout run
on separate nodes concurrently. Uses train_async.py and the fully_async_rollout
module so that weight updates do not block generation. Agent tasks are dispatched
to a Harbor-based agent server.

Tuned for long multi-turn code-agent trajectories (tool calls, reasoning, patches).
Default context budget is 64k tokens; batch sizes are kept small to fit long KV
cache during both rollout and training.

Qwen3.5-4B architecture: 32 layers (24 linear-attn + 8 full-attn), 16 attention
heads, hidden_size=2560, GQA (4 KV heads), attention-output-gate, vocab=248320.
TP must divide 16 (valid: 1, 2, 4, 8). Default split: 2 nodes training + 2 nodes
inference (configurable via --train-num-nodes), sized for a 4-node job.

SGLang note: Qwen3.5 requires rollout-num-gpus-per-engine=1 (TP>1 produces
garbage output on older SGLang builds). Unlike GLM Flash (8-GPU MoE engine +
DP-attention), dense 4B uses single-GPU engines; concurrency is KV-cache bound
and scales inversely with context length.

Data preparation (run separately before training):
    python download_and_process_data.py \\
        --input SWE-bench/SWE-bench_Verified \\
        --output /root/swe_train.jsonl \\
        --agent-name mini-swe-agent --split test

Usage:
    python run-qwen35-4B-agentic-async.py --num-nodes 4
    python run-qwen35-4B-agentic-async.py --num-nodes 4 --train-num-nodes 2
    python run-qwen35-4B-agentic-async.py --num-nodes 4 \\
        --agent-server-url http://ts-egress-aws-agent-server:8080
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Must run before `import miles`: launcher and Ray job both need this repo on sys.path.
SCRIPT_DIR = Path(__file__).resolve().parent
MILES_ROOT = SCRIPT_DIR.parent.resolve()
_FULLY_ASYNC_DIR = (MILES_ROOT / "examples" / "fully_async").resolve()
_miles_root_s = str(MILES_ROOT)
if _miles_root_s not in sys.path:
    sys.path.insert(0, _miles_root_s)
_pp = os.environ.get("PYTHONPATH", "")
if _miles_root_s not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{_miles_root_s}{os.pathsep}{_pp}" if _pp else _miles_root_s

import inspect

import typer

import miles.utils.external_utils.command_utils as U

FULLY_ASYNC_DIR = _FULLY_ASYNC_DIR

# Cluster-wide GPU-node ceiling for the ckpt-conversion job. Kept below the
# raw node count so ckpt conversion doesn't starve the rest of the cluster.
MAX_CONVERT_GPUS = 92

MODEL_DIR = "/home/yangchengyi/data/models"
CKPTS_DIR = "/home/yangchengyi/data/ckpts"
traj_dir = "/home/yangchengyi/data/trajs"
debug_msgs_dir = "/home/yangchengyi/data/debug_msgs"
data_dir = "/home/yangchengyi/data/datasets"
harbor_dir = "/home/yangchengyi/data/harbor/datasets/"
task = "swegym"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "qwen3.5-4B"
    num_nodes: int = 4
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    model_name: str = "Qwen3.5-4B"
    data_name: str = "deepmath-103k_miles"
    variant: str = f"{data_name}_{model_name}_deepmath_use_precision_aware"
    hf_checkpoint: str = f"{MODEL_DIR}/{model_name}"
    ref_load: str = f"{MODEL_DIR}/{model_name}_torch_dist"
    save_dir: str = f"{CKPTS_DIR}/{variant}/"
    # Directory to dump rollout + training traces (per-rollout .pt files). Empty
    # means default to ``<save_dir>/traces``; set to ``"disabled"`` to skip.
    save_traces_dir: str = f"{traj_dir}/{variant}"
    prompt_data: str = f"{data_dir}/{data_name}.jsonl"
    # Code-agent trajectories (multi-turn tool calls) can be very long.
    # max_seq_len: total session budget (prompt + all assistant turns + tool outputs).
    # rollout_max_response_len: max_new_tokens per single model call (one turn).
    max_seq_len: int = 65536
    rollout_max_response_len: int = 16384
    sglang_context_length: int = 65536
    sglang_mem_fraction: float = 0.80
    # Baseline per-engine slots at 8k ctx (Qwen3.5-9B colocate: 96 / 8 GPUs).
    sglang_short_ctx_slots_per_engine: int = 12
    sglang_context_ref_len: int = 8192
    sglang_decode_max_bs: int = 16
    save_interval: int = 50

    # Rollout / training batch sizing — kept small for 64k context memory budget.
    num_rollout: int = 3000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    over_sampling_batch_size: int = 8

    # Rollout precision
    rollout_fp8: bool = False
    rollout_health_check_first_wait: int = 3600

    # Agent settings
    agent_server_url: str = os.environ.get(
        "AGENT_SERVER_URL", "http://10.255.116.6:11000"
    )
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", f"{model_name}")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", f"{harbor_dir}/{task}")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # Disaggregated fully-async settings
    train_num_nodes: int = 2
    pause_generation_mode: Literal["in_place", "retract"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast", "p2p"] = "broadcast"
    accumulate_allreduce_grads_in_fp32: bool = True
    max_tokens_per_gpu: int = 4096

    
    """
    在 run-qwen35-4B-math.py 里去掉或设为 False：

    optimizer_cpu_offload: bool = False
    use_precision_aware_optimizer: bool = False
    # misc_args 里去掉 --grad-reduce-in-bf16
    """
    optimizer_cpu_offload: bool = False
    use_precision_aware_optimizer: bool = True

    # W&B settings
    wandb_key: str = os.environ.get(
        "WANDB_KEY",
        "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",
    )
    wandb_project: str = os.environ.get("WANDB_PROJECT", "miles-agentic")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = f'{variant}_{time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time() + 8 * 3600))}'
    disable_wandb_random_suffix: bool = True

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = variant


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
    multi_node = args.num_nodes > 1
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=multi_node,
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
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        # f"--max-seq-len {args.max_seq_len} "
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        # "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
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

    # Training parallelism for 4B + long context: TP=2, PP=1, CP=4 splits the
    # 64k sequence across GPUs. With 2 train nodes (16 GPUs): DP = 16 / 8 = 2.
    tp, pp, cp = 2, 1, 4
    dp = train_gpus // (tp * pp * cp)
    assert train_gpus % (tp * pp * cp) == 0, (
        f"train GPUs ({train_gpus}) must be divisible by TP*PP*CP ({tp * pp * cp})"
    )
    print(f"Training parallelism: TP={tp}, PP={pp}, CP={cp}, DP={dp}")
    max_tokens_per_gpu=args.rollout_max_response_len // cp
    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {pp} "
        f"--context-parallel-size {cp} "
        # "--expert-model-parallel-size 1 "
        # "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
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

    # SGLang: dense Qwen3.5-4B, 1 GPU/engine (TP=1 workaround).
    # Unlike GLM Flash MoE (world=8, attn_tp=4, decode_bs=256), dense engines are
    # KV-cache bound — slot count shrinks as context length grows.
    sglang_gpus_per_engine = 1
    num_engines = rollout_gpus // sglang_gpus_per_engine
    assert rollout_gpus % sglang_gpus_per_engine == 0, (
        f"rollout GPUs ({rollout_gpus}) must be divisible by "
        f"sglang_gpus_per_engine ({sglang_gpus_per_engine})"
    )
    print(f"Inference: {num_engines} engines x {sglang_gpus_per_engine} GPU/engine")

    ctx_scale = max(1, args.sglang_context_length // args.sglang_context_ref_len)
    sglang_per_engine_slots = max(1, args.sglang_short_ctx_slots_per_engine // ctx_scale)
    sglang_chunked_prefill_size = min(
        max(args.sglang_context_length // 16, 4096),
        16384,
    )
    sglang_decode_max_bs = min(args.sglang_decode_max_bs, sglang_per_engine_slots * 4)
    # Per-engine caps passed to each SGLang server; miles scales client semaphore
    # cluster-wide as: server_concurrency * rollout_gpus // gpus_per_engine.
    sglang_max_running_requests = sglang_per_engine_slots
    sglang_server_concurrency = sglang_per_engine_slots
    cluster_max_connections = (
        sglang_server_concurrency * rollout_gpus // sglang_gpus_per_engine
    )
    print(
        f"SGLang (dense): ctx={args.sglang_context_length}, ctx_scale={ctx_scale}, "
        f"per_engine_slots={sglang_per_engine_slots}, "
        f"max_running/engine={sglang_max_running_requests}, "
        f"chunked_prefill={sglang_chunked_prefill_size}, "
        f"decode_max_bs={sglang_decode_max_bs}, "
        f"cluster_connections={cluster_max_connections}"
    )

    sglang_p2p_extra = ""
    if args.update_weight_transfer_mode == "p2p":
        sglang_p2p_extra = "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction} "
        # f"--sglang-context-length {args.sglang_context_length} "
        # f"--sglang-max-running-requests {sglang_max_running_requests} "
        # f"--sglang-chunked-prefill-size {sglang_chunked_prefill_size} "
        # f"--sglang-server-concurrency {sglang_server_concurrency} "
        "--sglang-tool-call-parser qwen3_coder "
        # "--use-sglang-tool-choice-required "
        "--sglang-grammar-backend xgrammar "
        "--use-miles-router "
        "--sglang-router-port 31000 "
        f"{sglang_p2p_extra}"
    )
    sglang_extra_env_vars: dict[str, str] = {
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    }

    # agent_args = (
    #     "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
    #     "--custom-agent-function-path swe_agent_function.run "
    #     "--custom-rm-path generate.reward_func "
    #     "--tito-model qwen35 "
    #     "--use-session-server "
    #     "--session-server-port 30000 "
    #     "--tito-allowed-append-roles user tool "
    #     f"--session-debug-dump-dir {os.path.join(debug_msgs_dir, args.wandb_run_name)} "
    # )

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
        # f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    # K8s: Ray cluster is started by ray_init_simple.sh; only ray job submit here.
    os.environ.setdefault("MILES_SCRIPT_EXTERNAL_RAY", "true")
    if not os.environ.get("MASTER_ADDR") and os.environ.get("POD_IP"):
        os.environ["MASTER_ADDR"] = os.environ["POD_IP"]
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    ray_dashboard_port = os.environ.get("RAY_DASHBOARD_PORT", "8265")
    os.environ.setdefault("RAY_ADDRESS", f"http://{master_addr}:{ray_dashboard_port}")
    print(
        f"External Ray: MASTER_ADDR={master_addr}, RAY_ADDRESS={os.environ['RAY_ADDRESS']} "
        f"(set MILES_SCRIPT_EXTERNAL_RAY=0 to let the script start Ray locally)"
    )

    miles_root = MILES_ROOT
    extra_env_vars = {
        "PYTHONPATH": f"{miles_root}:{args.megatron_path}:{SCRIPT_DIR}:{FULLY_ASYNC_DIR}",
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
        print(f'{extra_env_vars["MILES_ROUTER_EXTERNAL_HOST"]=}')
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip
        print(f'{extra_env_vars["MILES_HOST_IP"]=}')

    execute_train_kw = {
        "train_args": train_args,
        "config": args,
        "num_gpus_per_node": args.num_gpus_per_node,
        "megatron_model_type": args.megatron_model_type,
        "train_script": str(miles_root / "train_async.py"),
        "megatron_path": args.megatron_path,
        "extra_env_vars": extra_env_vars,
    }
    if "skip_cleanup" in inspect.signature(U.execute_train).parameters:
        execute_train_kw["skip_cleanup"] = True
    U.execute_train(**execute_train_kw)


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
