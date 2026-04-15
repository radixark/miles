"""GLM-4.7 Full (355B-A32B) reasoning training with GSM8K data.

Debug script: uses math (GSM8K) data instead of agentic tool use to verify
that the training pipeline produces nonzero rewards and learns successfully.

Same model architecture and parallelism as run-glm47-full.py.
Targets 16 x 8-GPU H200 nodes (sci-h200).

Usage:
    python run-glm47-reasoning.py --num-nodes 16
    python run-glm47-reasoning.py --num-nodes 16 --rollout-fp8
    python run-glm47-reasoning.py --num-nodes 16 --skip-prepare
    python run-glm47-reasoning.py --num-nodes 16 --mode debug_rollout_only
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
    megatron_model_type: str = "glm4.5-355B-A32B"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    model_name: str = "GLM-4.7"
    hf_checkpoint: str = "/models/zai-org/GLM-4.7"
    ref_load: str = "/models/zai-org/GLM-4.7_torch_dist"
    save_dir: str = "/root/GLM-4.7-Full_reasoning/"
    prompt_data: str = "/root/datasets/gsm8k/train.parquet"
    rollout_max_response_len: int = 1024

    # Rollout precision
    rollout_fp8: bool = False

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "glm47-full-reasoning")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm47-full-gsm8k"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "glm47-full-gsm8k"


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


def _convert_hf_to_fp8(args: ScriptArgs):
    """Convert HF bf16 checkpoint to block-wise FP8 for SGLang rollout."""
    fp8_dir = f"{args.hf_checkpoint}-FP8"
    if Path(fp8_dir).exists():
        print(f"FP8 checkpoint already exists at {fp8_dir}, skipping conversion.")
        return
    U.exec_command(
        "python tools/convert_hf_to_fp8.py "
        f"--model-dir {args.hf_checkpoint} "
        f"--save-dir {fp8_dir} "
        "--strategy block --block-size 128 128 "
        "--max-workers 4"
    )


def prepare(args: ScriptArgs):
    """Download GSM8K data and convert HF checkpoint to torch_dist format."""
    # Download GSM8K dataset
    U.hf_download_dataset("zhuzilin/gsm8k")

    # Convert checkpoint (multinode for 355B)
    # The conversion tool requires world_size <= num_layers (92 for this model).
    max_convert_nodes = 92 // args.num_gpus_per_node  # 11 for 8 GPUs/node
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

    if args.rollout_fp8:
        _convert_hf_to_fp8(args)


def execute(args: ScriptArgs):
    hf_checkpoint = f"{args.hf_checkpoint}-FP8" if args.rollout_fp8 else args.hf_checkpoint
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        "--save-interval 100 "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-temperature 0.8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 64 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--skip-eval-before-train "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    # Training parallelism: TP=4, PP=2, EP chosen as largest divisor of 160 that fits.
    tp, pp = 4, 2
    total_gpus = args.num_nodes * args.num_gpus_per_node
    dp = total_gpus // (tp * pp)
    assert total_gpus % (tp * pp) == 0, f"total GPUs ({total_gpus}) must be divisible by TP*PP ({tp * pp})"
    num_experts = 160
    ep = max(d for d in range(1, dp + 1) if num_experts % d == 0)

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
        "--max-tokens-per-gpu 2048 "
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

    # SGLang: 4 nodes/engine with full EP + DP-attention.
    # 355B across 32 GPUs → ~22GB/GPU (bf16) or ~11GB/GPU (FP8) for weights,
    # leaving plenty for KV cache. EP=32 with 160 experts → 5 experts/GPU.
    # DP-attention keeps attention within a single node (attn_tp=8).
    sglang_nodes_per_engine = min(4, args.num_nodes)
    sglang_world_size = sglang_nodes_per_engine * args.num_gpus_per_node
    assert (
        total_gpus % sglang_world_size == 0
    ), f"total GPUs ({total_gpus}) must be divisible by sglang_world_size ({sglang_world_size})"
    sglang_decode_max_bs = 256
    sglang_attn_tp_size = min(args.num_gpus_per_node, sglang_world_size)
    sglang_attn_dp_size = sglang_world_size // sglang_attn_tp_size
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.80 "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-enable-dp-attention "
        f"--sglang-dp-size {sglang_attn_dp_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-moe-a2a-backend deepep "
        "--sglang-deepep-mode low_latency "
        f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
        f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
    )
    if args.rollout_fp8:
        sglang_args += "--sglang-moe-runner-backend deep_gemm "
    sglang_extra_env_vars = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": f"{sglang_decode_max_bs}",
    }

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {total_gpus} "
        "--use-fault-tolerance "
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
        f"{eval_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "NCCL_NVLS_ENABLE": "0",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        **sglang_extra_env_vars,
    }

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
