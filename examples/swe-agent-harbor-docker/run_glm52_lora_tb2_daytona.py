"""GLM-5.2 744B-A40B **LoRA** agentic launcher: Terminal-Bench-2 style tasks on Daytona sandboxes.

Combines two paths that previously did not meet:
  * the agentic Harbor path from ``run.py`` (session server + TITO + terminus-2), and
  * the GLM-5.2 MoE/MLA/DSA LoRA path from ``scripts/run_glm5_2_744b_a40b_lora.py``.

The 744B base does not fit on one node (bf16 ~1403 GiB vs 1123 GiB of HBM on 8x H200),
so the trainer is sharded over ``--num-nodes`` (EP spans the whole world, TP stays
intra-node) and the rollout is served from the fp8 checkpoint. Ray must already be up
across every node, so set ``MILES_SCRIPT_EXTERNAL_RAY=1``.

The two DSA backends differ by a sequence-length ceiling, so prefer ``tilelang``:

``megatron`` is a naive dense reference, not a memory-efficient one. ``unfused_dsa_fn``
materializes the full fp32 ``[b, np, S, S]`` score matrix and applies DSA sparsity as an
additive ``-inf`` mask, so the top-k buys no memory and no FLOPs. With 78 layers and 8
heads per rank at TP=8 that retains roughly ``3744 * S**2`` bytes: 59 GiB at S=4096 and
234 GiB at S=8192. It also requires the bshd query layout, which forbids
``--use-dynamic-batch-size`` (hence ``--micro-batch-size 1``) and rules out activation
recompute, because GLM-5.2 shares DSA top-k across layers via ``packed_seq_params``,
which bshd does not carry, so a recomputed skip layer would read a stale anchor top-k.
Together these cap it near S=4096, and since the cost is quadratic in S, no larger GPU
moves that ceiling much -- 2x the memory buys only sqrt(2) the sequence length.

``tilelang`` never materializes the matrix (it is O(S * topk)) and uses the thd layout,
whose ``packed_seq_params`` is closure-captured by Megatron's checkpoint
``custom_forward``, re-enabling activation recompute. It trains S=16384 on 4 nodes with
room to spare.

Usage (4 nodes x 8 H200, ray already running):
  MILES_SCRIPT_EXTERNAL_RAY=1 python run_glm52_lora_tb2_daytona.py \\
      --num-nodes 4 --prompt-data /root/tb2_train.jsonl
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

# Attention + MLA only, EXCLUDING the DSA indexer (wq_b/wk/weights_proj) — on
# tilelang the indexer adapter gets no gradient at all — and EXCLUDING the MLP/MoE
# leaves (gate_proj/up_proj/down_proj): this is the MoE-LoRA-off ablation, so all
# experts (routed + shared) and the dense MLP stay frozen.
_DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "glm5.2-744B-A40B_lora"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    hf_checkpoint: str = "/models/zai-org/GLM-5.2"
    # Rollout-side fp8 checkpoint; the trainer stays bf16.
    fp8_rollout_checkpoint: str = "/models/zai-org/GLM-5.2-FP8"
    # Must be shared across nodes: every rank writes its dist-checkpoint shard here,
    # and the generated sglang config is read by engine actors on every node.
    save_dir: str = "/root/GLM-5.2_lora_tb2/"
    save_traces_dir: str = ""
    prompt_data: str = "/root/tb2_train.jsonl"

    # Sequence budget: --max-seq-len caps the whole session (prompt + every
    # completion + every env response); --rollout-max-response-len caps one turn.
    max_seq_len: int = 65536
    rollout_max_response_len: int = 8192
    # Serving window, deliberately independent of max_seq_len. max_seq_len only trims
    # what the trainer keeps, and the megatron DSA indexer is O(S^2) in fp32 so it has
    # to stay small; the agent still needs the full window or its very first request is
    # rejected for exceeding the context.
    sglang_context_length: int = 65536

    # Training settings
    num_rollout: int = 200
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 10
    lr: str = "3e-5"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    # Required for true on-policy under colocate (OFF -> KL ~1.0 vs ~1e-4).
    lora_base_cpu_backup: bool = True
    # Expert-LoRA layout flag; irrelevant (and left off) with no expert leaves in
    # --target-modules.
    experts_shared_outer_loras: bool = False

    # GLM-5.2 specifics. megatron is dense O(S**2) and caps near S=4096, so agentic
    # sequence lengths need tilelang; see the module docstring.
    dsa_attention_backend: Literal["megatron", "tilelang"] = "tilelang"
    # R3 rollout routing replay (arxiv 2510.11370)
    use_r3: bool = True

    # Rollout engine.
    # Under colocate + LoRA sglang mirrors each GPU's weight shard into host RAM
    # (enable_weights_cpu_backup), costing (fp8 ckpt / this) * gpus_per_node per node:
    # against the 744B fp8 ckpt, 8 costs ~704 GiB/node and 16 halves that. 8 leaves too
    # little room for the concurrent Megatron bridge load and OOMs the pod cgroup.
    fp8_rollout_gpus_per_engine: int = 16
    sglang_mem_fraction_static: float = 0.85
    # The paused actor is ~50 GiB/rank, so a pinned host copy costs ~400 GiB/node on top of
    # sglang's own backup and OOMs a 2 TiB node during offload. Spill it to node-local disk
    # instead; host use then stays at one chunk per rank. Must not be tmpfs.
    offload_train_disk_dir: str = "/scratch/miles_train_offload"
    # sglang's own default (csgmv) crashes the DSA MoE-LoRA rollout under dp-attention
    sglang_lora_backend: str = "triton"

    # Agent settings
    agent_server_url: str = os.environ.get("AGENT_SERVER_URL", "http://localhost:8080")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    # sgl-router binds with a Rust SocketAddr parse, so this MUST be a numeric IP.
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "my-wandb-project")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm52-lora-tb2-daytona"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9091
    prometheus_run_name: str = "glm52-lora-tb2-daytona"


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "MegatronTrain"]
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


def _parallel_args(args: ScriptArgs) -> str:
    """TP = num_gpus_per_node (intra-node, so the TP all-reduce keeps to NVLink),
    EP = the whole world, ETP 1. Megatron requires EP * ETP == TP * DP, which holds
    for any node count with PP = CP = 1.
    """
    ngpu = args.num_gpus_per_node
    world_size = args.num_nodes * ngpu
    # megatron's unfused DSA core-attention takes a 4D query, so bshd; bshd in turn
    # forbids --use-dynamic-batch-size, so microbatches are single sequences.
    qkv_format = "thd" if args.dsa_attention_backend == "tilelang" else "bshd"
    # Only thd carries packed_seq_params, and the cross-layer DSA top-k holder rides on
    # it, so Megatron's checkpoint custom_forward closure-captures the right anchor top-k
    # at recompute time. bshd falls back to a thread-local the closure never sees, and
    # megatron-bridge asserts on that combination rather than emit stale gradients.
    recompute = (
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        if qkv_format == "thd"
        else ""
    )
    return (
        f"--tensor-model-parallel-size {ngpu} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {world_size} "
        "--expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} "
        "--micro-batch-size 1 "
        f"{recompute}"
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )


def _sglang_args(args: ScriptArgs) -> str:
    world_size = args.num_nodes * args.num_gpus_per_node
    engine = min(args.fp8_rollout_gpus_per_engine, world_size)
    # Keep the running batch inside a captured graph; the graph only has to cover
    # rollout_batch_size * n_samples_per_prompt concurrent sessions.
    max_bs = 64
    return (
        f"--rollout-num-gpus-per-engine {engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        # dp-attention is off on purpose: its scheduler loop runs per-iteration
        # collectives (control-msg gloo broadcast, mlp-sync all_gather) that must stay
        # in lockstep forever; colocate weight syncs (abort/retract churn with the
        # broadcast, resume barriers with local-control-broadcast) desync them and
        # deadlock the engines. Plain TP + EP MoE has no such collectives.
        f"--sglang-ep-size {engine} "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-decode-backend flashmla_kv "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-page-size 64 "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        f"--sglang-context-length {args.sglang_context_length} "
        f"--sglang-cuda-graph-max-bs {max_bs} --sglang-max-running-requests {max_bs} "
        f"--sglang-chunked-prefill-size {min(8192, 2048 * engine)} "
        "--sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
        # required: without it sglang miscounts the gate_up slices -> engine-init crash
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-router-port 31001 "
    )


def _write_sglang_fp8_config(args: ScriptArgs) -> str:
    """Serve the fp8 ckpt while the trainer stays bf16. update_weights stays on so the
    per-step LoRA sync reaches the engine; the bf16 base sync is already skipped under
    colocate + cpu backup."""
    path = f"{args.save_dir.rstrip('/')}/sglang_fp8_rollout.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "sglang:\n"
            "  - name: default\n"
            f"    model_path: {args.fp8_rollout_checkpoint}\n"
            "    update_weights: true\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            # total GPUs for the group, not per engine: under --colocate the rollout
            # spans the same world as the actor, split into world/engine engines
            f"        num_gpus: {args.num_nodes * args.num_gpus_per_node}\n"
        )
    return path


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        "--megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
    )
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

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

    # No --ref-load: under LoRA the reference policy is the base model with the
    # adapter disabled, so a separate 744B reference checkpoint is unnecessary.
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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

    # routing replay only: --use-rollout-indexer-replay is debug-only and its
    # ~78-128 GB/rank host buffer OOMs the colocate pod
    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30001 "
        "--session-server-workers 32 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    misc_args += (
        "--offload-train-target disk "
        f"--offload-train-disk-dir {args.offload_train_disk_dir} "
        "--offload-train-disk-chunk-mb 256 "
    )

    traces_dir = args.save_traces_dir or f"{args.save_dir.rstrip('/')}/traces"
    if traces_dir != "disabled":
        misc_args += f"--dump-details {traces_dir} --use-miles-dashboard "

    # train/entropy_loss is a hardcoded 0.0 unless this is set, and a falling entropy is the
    # earliest warning of policy collapse on a long agentic run.
    misc_args += "--observe-training-entropy "

    # Under bf16 there is no grad scaler, so Megatron's prepare_grads() returns found_inf=False
    # unconditionally and a non-finite grad norm reaches the step, where clipping by
    # clip/(norm + eps) writes NaN into every adapter tensor. This flag routes the step through
    # miles' own guard instead, which skips the offending step and leaves the weights intact.
    misc_args += "--no-check-for-nan-in-loss-and-grad "

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

    sglang_args = _sglang_args(args) + f"--sglang-config {_write_sglang_fp8_config(args)} "

    train_args = (
        f"{ckpt_args}"
        f"{lora_args}"
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{r3_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{_parallel_args(args)}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        # GLM-5 DSA indexer uses interleaved RoPE; a mismatch garbles long sequences
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "SGLANG_NSA_FORCE_MLA": "1",
        # Variable-length agentic batches fragment the allocator pool until NCCL's own
        # cudaMalloc (outside torch's pool) finds nothing free during the LoRA grad
        # all-reduce; gc_threshold + max_split_size keep the pool releasable.
        # expandable_segments:True, the usual answer, breaks torch_memory_saver under
        # colocate. Do NOT add per_process_memory_fraction (torch >= 2.10) here: these
        # env vars reach every actor in the ray job, so under --colocate the rollout
        # engines would inherit the cap and OOM below --sglang-mem-fraction-static.
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
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
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    execute(args)


if __name__ == "__main__":
    typer.run(main)
