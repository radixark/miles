"""
Multi-model 4-node profiling script for RDMA/NCCL weight transfer.

Runs 6 model configs sequentially (NCCL, RDMA, and/or RDMA shared-buffer modes),
writing timer logs under $MILES_LOG_DIR/4node-profile/<model>/<mode>/.
Runs 13 steps; average the last 10 for stable profiling numbers.

Usage:
    python test_weight_transfer_moe_multinode.py \
        --multinode --head-node-ip <IP> --node-rank <RANK> --nnodes 4 \
        [--mode nccl|rdma|rdma-shared|all] \
        [--models llama3,glm4,glm45-air,moonlight,qwen3-30b,qwen3-32b]
"""

import os
import time
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U
from miles.utils.timer import log_experiment_start

GPUS_PER_NODE = 8


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    key: str  # short identifier for CLI and log folders
    model_name: str  # HF model directory name
    hf_repo: str  # HF download path
    model_type: str  # megatron_model_type (maps to scripts/models/<type>.sh)
    # Training parallelism
    train_tp: int
    train_ep: int
    train_pp: int = 1
    train_cp: int = 1
    train_etp: int = 1
    # Rollout parallelism (2 engines of 8 GPUs each)
    sglang_tp: int = 8
    sglang_ep: int = 1
    # Decoder split (only needed when PP > 1)
    decoder_last_pipeline_num_layers: int | None = None
    # Model-specific rotary base override
    rotary_base: str | None = None
    # Extra training flags (e.g. deepep, flex dispatcher)
    extra_train_flags: str = ""


MODELS: dict[str, ModelConfig] = {
    "glm4": ModelConfig(
        key="glm4",
        model_name="GLM-Z1-9B-0414",
        hf_repo="zai-org/GLM-Z1-9B-0414",
        model_type="glm4-9B",
        train_tp=2,
        train_ep=1,
        train_cp=2,
    ),
    "moonlight": ModelConfig(
        key="moonlight",
        model_name="Moonlight-16B-A3B-Instruct",
        hf_repo="moonshotai/Moonlight-16B-A3B-Instruct",
        model_type="moonlight",
        train_tp=2,
        train_ep=8,
        sglang_ep=8,
    ),
    "qwen3-30b": ModelConfig(
        key="qwen3-30b",
        model_name="Qwen3-30B-A3B",
        hf_repo="Qwen/Qwen3-30B-A3B",
        model_type="qwen3-30B-A3B",
        train_tp=4,
        train_ep=8,
        sglang_ep=8,
        rotary_base="1000000",
    ),
    "qwen3-32b": ModelConfig(
        key="qwen3-32b",
        model_name="Qwen3-32B",
        hf_repo="Qwen/Qwen3-32B",
        model_type="qwen3-32B",
        train_tp=8,
        train_ep=1,
    ),
}

ALL_MODEL_KEYS = list(MODELS.keys())


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["nccl", "rdma", "rdma-shared", "all"] = "all"
    models: str = ",".join(ALL_MODEL_KEYS)  # comma-separated model keys
    skip_validation: bool = False
    # Multi-node settings
    multinode: bool = True
    head_node_ip: str | None = None
    node_rank: int = 0
    nnodes: int = 4
    # Resource split: 16 train + 16 rollout (4 nodes)
    num_train_gpus: int = 2 * GPUS_PER_NODE
    num_rollout_gpus: int = 2 * GPUS_PER_NODE
    # Misc
    bucket_size: float = 1.0
    released_mc_transfer_timeout: bool = False
    no_save_optim: bool = False
    wait_after: bool = False
    enable_nccl_nvls: bool = False

    def validate(self):
        assert self.multinode, "This script is for multi-node profiling only"
        assert self.num_train_gpus % GPUS_PER_NODE == 0
        assert self.num_rollout_gpus % GPUS_PER_NODE == 0
        assert self.num_train_gpus + self.num_rollout_gpus == self.nnodes * GPUS_PER_NODE

    def selected_models(self) -> list[ModelConfig]:
        keys = [k.strip() for k in self.models.split(",")]
        out = []
        for k in keys:
            if k not in MODELS:
                raise ValueError(f"Unknown model key '{k}'. Choose from: {ALL_MODEL_KEYS}")
            out.append(MODELS[k])
        return out

    def selected_modes(self) -> list[str]:
        if self.mode == "all":
            return ["nccl", "rdma", "rdma-shared"]
        else:
            return [self.mode]


# ---------------------------------------------------------------------------
# Prepare: download + convert
# ---------------------------------------------------------------------------
def prepare(args: ScriptArgs, cfg: ModelConfig):
    if args.node_rank == 0:
        U.exec_command("mkdir -p /root/models /root/datasets")
        U.exec_command(f"hf download {cfg.hf_repo} --local-dir /root/models/{cfg.model_name}")
        U.hf_download_dataset("zhuzilin/dapo-math-17k")
        U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=cfg.model_name,
        megatron_model_type=cfg.model_type,
        num_gpus_per_node=GPUS_PER_NODE,
        multinode=True,
        master_addr=args.head_node_ip,
        nnodes=args.nnodes,
        dir_dst="/root/multinode",
        node_rank=args.node_rank,
        decoder_last_pipeline_num_layers=cfg.decoder_last_pipeline_num_layers,
    )


# ---------------------------------------------------------------------------
# Execute one (model, mode) pair
# ---------------------------------------------------------------------------
def execute(args: ScriptArgs, cfg: ModelConfig, mode: str, base_log_dir: str, is_last_mode: bool = True):
    is_rdma = mode in ("rdma", "rdma-shared")

    run_log_dir = f"{base_log_dir}/4node-profile/{cfg.key}/{mode}"
    os.makedirs(run_log_dir, exist_ok=True)
    os.environ["MILES_LOG_DIR"] = run_log_dir

    log_experiment_start(
        {
            "mode": mode,
            "model": cfg.model_name,
            "model_type": cfg.model_type,
            "num_train_gpus": args.num_train_gpus,
            "num_rollout_gpus": args.num_rollout_gpus,
            "train_tp": cfg.train_tp,
            "train_ep": cfg.train_ep,
            "train_pp": cfg.train_pp,
            "train_cp": cfg.train_cp,
            "train_etp": cfg.train_etp,
            "sglang_tp": cfg.sglang_tp,
            "sglang_ep": cfg.sglang_ep,
            "multinode": args.multinode,
            "nnodes": args.nnodes,
            "node_rank": args.node_rank,
        }
    )

    # --- Checkpoint ---
    ckpt_args = (
        f"--hf-checkpoint /root/models/{cfg.model_name}/ " f"--ref-load /root/multinode/{cfg.model_name}_torch_dist/ "
    )
    if args.no_save_optim:
        ckpt_args += "--no-save-optim "

    # --- Rollout ---
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 13 --rollout-batch-size 4 --n-samples-per-prompt 4 "
        "--rollout-max-response-len 100 --rollout-temperature 0.8 "
        "--global-batch-size 16 --balance-data "
    )

    # --- Training parallelism ---
    perf_args = (
        f"--tensor-model-parallel-size {cfg.train_tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {cfg.train_pp} "
        f"--context-parallel-size {cfg.train_cp} "
        f"--expert-model-parallel-size {cfg.train_ep} "
        f"--expert-tensor-parallel-size {cfg.train_etp} "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    )
    if cfg.decoder_last_pipeline_num_layers is not None:
        perf_args += f"--decoder-last-pipeline-num-layers {cfg.decoder_last_pipeline_num_layers} "

    # --- Eval ---
    eval_args = (
        "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 --eval-max-response-len 16384 --eval-top-p 0.7 "
    )

    # --- GRPO ---
    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 4e-4 "
    )

    # --- Optimizer ---
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    # --- SGLang: 2 engines x 8 GPUs, no DP, with router ---
    sglang_args = (
        f"--rollout-num-gpus-per-engine {cfg.sglang_tp} "
        f"--rollout-num-gpus {args.num_rollout_gpus} "
        "--sglang-mem-fraction-static 0.8 "
        f"--sglang-ep-size {cfg.sglang_ep} "
        "--sglang-cuda-graph-bs 1 2 4 8 16 "
        "--use-miles-router "
    )
    if cfg.sglang_ep > 1:
        sglang_args += "--sglang-enable-dp-attention --sglang-enable-dp-lm-head "
    if is_rdma:
        sglang_args += "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
    if args.skip_validation:
        sglang_args += "--sglang-load-format dummy "

    # --- Misc ---
    mem = int(args.bucket_size * 1024 * 1024 * 1024) if is_rdma else (4 * 1024 * 1024 * 1024)
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_train_gpus // GPUS_PER_NODE} "
        f"--actor-num-gpus-per-node {GPUS_PER_NODE} "
        f"--update-weight-buffer-size {mem} "
    )
    if not args.skip_validation:
        misc_args += "--check-weight-update-equal "
    if is_rdma:
        misc_args += "--update-weight-transfer-mode rdma "
    if mode == "rdma-shared":
        misc_args += "--rdma-shared-buffer "
    misc_args += cfg.extra_train_flags

    # --- Assemble ---
    train_args = (
        f"{ckpt_args} {rollout_args} {eval_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} {sglang_args} {misc_args}"
    )

    # Worker nodes start late to give head node time
    if args.node_rank > 0:
        time.sleep(20)

    if cfg.rotary_base:
        os.environ["MODEL_ARGS_ROTARY_BASE"] = cfg.rotary_base

    mc_transfer_timeout = "300" if args.released_mc_transfer_timeout else "30"
    num_gpus = args.num_train_gpus + args.num_rollout_gpus

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=GPUS_PER_NODE,
        megatron_model_type=cfg.model_type,
        train_script="train.py",
        extra_env_vars={
            "MC_TRANSFER_TIMEOUT": mc_transfer_timeout,
            "RAY_DEBUG": "1",
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1" if args.enable_nccl_nvls else "0",
            "MILES_LOG_DIR": run_log_dir,
        },
        multinode=args.multinode,
        is_head_node=args.node_rank == 0,
        num_gpus=num_gpus,
    )

    if args.node_rank > 0 and args.wait_after:
        if is_last_mode:
            # Only sleep on the very last mode; intermediate modes are
            # synchronised by the head-node's blocking `ray job submit`.
            time.sleep(800 if mode == "nccl" else 3600)
        else:
            # For intermediate modes, wait for the head node's ray job to
            # finish by polling until the Ray GCS is unreachable (head node
            # did `ray stop`).  Then stop the local Ray daemon so we can
            # rejoin a fresh cluster for the next mode.
            import ray

            while True:
                try:
                    ray.init(address="auto", ignore_reinit_error=True)
                    ray.available_resources().get("GPU", 0)
                    ray.shutdown()
                    # If the head-node has torn down the cluster the init
                    # call above will raise.  While it still succeeds the
                    # job is still running (or the head hasn't killed Ray
                    # yet) – keep waiting.
                    time.sleep(5)
                except Exception:
                    break
            # Stop local Ray so we can rejoin the next cluster
            U.exec_command("ray stop --force; pkill -9 ray; true")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@U.dataclass_cli
def main(args: ScriptArgs):
    args.validate()
    modes = args.selected_modes()
    model_cfgs = args.selected_models()

    # Save original log dir before the loop to prevent nesting
    base_log_dir = os.environ.get("MILES_LOG_DIR", "/root")

    for ci, cfg in enumerate(model_cfgs):
        prepare(args, cfg)
        for mi, mode in enumerate(modes):
            is_last = (ci == len(model_cfgs) - 1) and (mi == len(modes) - 1)
            print(f"\n{'='*60}")
            print(f"  Running: {cfg.key} / {mode}")
            print(f"{'='*60}\n")
            execute(args, cfg, mode, base_log_dir, is_last_mode=is_last)


if __name__ == "__main__":
    typer.run(main)
