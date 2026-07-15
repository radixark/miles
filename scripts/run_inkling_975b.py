"""
Inkling 975B training script.

Supports:
  - Inkling          66-layer 975B MoE (frozen vision/audio towers optional).
                          Verified profiles: 16 nodes x 4 GPUs (TP4 PP4 EP16) and
                          12 nodes x 4 GPUs (TP4 PP3 EP16) on GB300.
  - Inkling-4layer   4-layer slice for single-node smoke testing.

Train modes:
  - full   Full-parameter GRPO. Optimizer state streams through node-local NVMe
           (--optimizer-state-nvme-dir, GPU-stepped, one bucket resident) and the
           paused training actor spills to disk via torch_memory_saver.
  - lora   LoRA r=32 all-linear. Adapter-only weight sync; plain fp32 Adam;
           engine serves the adapter natively (triton backend, virtual experts).

Tasks:
  - dapo_math  text (dapo-math-17k), chat template, aime25 eval.
  - geo3k      vision (geo3k), structured-message rendering, frozen towers,
               geo3k-val eval.

Usage patterns:

  1. Train on pre-staged checkpoints:
       python scripts/run_inkling_975b.py train \
           --model-name Inkling --train-mode full --task dapo_math \
           --num-nodes 16 --num-gpus-per-node 4

  2. Individual steps (rsync shared -> node-local NVMe, then train):
       python scripts/run_inkling_975b.py prepare-cp --model-name Inkling
       python scripts/run_inkling_975b.py train --model-name Inkling ...

  3. One-shot (prepare-cp when model_local_dir differs, then train):
       python scripts/run_inkling_975b.py full-train --model-name Inkling ...
"""

from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MODEL_NUM_LAYERS = {
    "Inkling": 66,
    "Inkling-4layer": 4,
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["Inkling", "Inkling-4layer"] = "Inkling"

    train_mode: Literal["full", "lora"] = "full"
    task: Literal["dapo_math", "geo3k"] = "dapo_math"
    enable_eval: bool = False
    num_rollout: int = 100

    hf_checkpoint: str | None = None
    torch_dist: str | None = None
    # Defaults to torch_dist. Set explicitly when shared NFS -> per-node local NVMe copy is needed.
    torch_dist_local: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    save_dir: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    # performance configs
    num_gpus_per_node: int = 4
    lr: float | None = None
    rollout_max_response_len: int = 4096
    sglang_context_length: int = 8192
    optimizer_nvme_dir: str = "/tmp/opt_offload"
    train_offload_disk_dir: str = "/tmp/train_offload"
    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)

    # LoRA configs
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_adapter_path: str | None = None

    enable_r3: bool = True

    # pass any extra sglang/miles/megatron args through `--extra-args '--your-arg'`
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.torch_dist is None:
            self.torch_dist = f"{self.model_dir}/{self.model_name}_torch_dist"
        if self.torch_dist_local is None:
            self.torch_dist_local = self.torch_dist
        if self.lr is None:
            self.lr = 5e-6 if self.train_mode == "lora" else 1e-6
        self.colocate = True
        self.actor_num_nodes = self.num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node

    @property
    def num_layers(self):
        return _MODEL_NUM_LAYERS[self.model_name]

    @property
    def is_mm(self):
        return self.task == "geo3k"


def _get_parallel_config(args: ScriptArgs) -> str:
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    total_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.model_name == "Inkling-4layer" and args.actor_num_nodes == 1:
        return (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )

    if args.actor_num_gpus_per_node == 4:
        if total_gpus == 64:  # 16 nodes x 4 GPUs: 66 = 3x17 + 15
            return (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 4 "
                "--decoder-last-pipeline-num-layers 15 "
                "--expert-model-parallel-size 16 "
                "--expert-tensor-parallel-size 1 "
            )
        if total_gpus == 48:  # 12 nodes x 4 GPUs: 66 = 3x22
            return (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 3 "
                "--expert-model-parallel-size 16 "
                "--expert-tensor-parallel-size 1 "
            )

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} GPUs. "
        f"Please specify your parallel config in `run_inkling_975b._get_parallel_config`."
    )


def _train(args: ScriptArgs):
    print(
        f"running {args.model_name} {args.train_mode}/{args.task} on "
        f"{args.num_nodes} nodes x {args.num_gpus_per_node} GPUs (colocate)"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {args.torch_dist_local} "
        "--model-name inkling "
        "--megatron-to-hf-mode raw "
        "--no-load-optim --no-load-rng --finetune "
    )
    if args.save_dir is not None:
        ckpt_args += f"--save {args.save_dir}/{args.run_id}/checkpoints --save-interval 10 "

    rollout_args = (
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
        "--balance-data "
    )
    eval_args = ""
    match args.task:
        case "dapo_math":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl " "--apply-chat-template "
            )
            if args.enable_eval:
                eval_args = (
                    "--eval-interval 5 "
                    f"--eval-prompt-data aime25 {args.data_dir}/aime-2025/aime-2025.jsonl "
                    "--eval-input-key prompt --eval-label-key label "
                    "--n-samples-per-eval-prompt 1 --eval-temperature 1 "
                    f"--eval-max-response-len {args.rollout_max_response_len} "
                )
        case "geo3k":
            # structured message lists rendered by the Inkling processor - no chat template
            rollout_args += f"--prompt-data {args.data_dir}/geo3k/geo3k_train.jsonl "
            if args.enable_eval:
                eval_args = (
                    "--eval-interval 5 "
                    f"--eval-prompt-data geo3kval {args.data_dir}/geo3k/geo3k_val.jsonl "
                    "--eval-input-key prompt --eval-label-key label "
                    "--n-samples-per-eval-prompt 1 --eval-temperature 1 "
                    f"--eval-max-response-len {args.rollout_max_response_len} "
                )

    grpo_args = (
        "--advantage-estimator grpo "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--eps-clip-c 3.0 "
        "--use-tis "
    )
    if args.enable_r3:
        grpo_args += "--use-rollout-routing-replay "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-distributed-optimizer "
        "--accumulate-allreduce-grads-in-fp32 "
        "--no-check-for-nan-in-loss-and-grad "
    )

    perf_args = _get_parallel_config(args)
    perf_args += "--recompute-granularity full " "--recompute-method uniform " "--recompute-num-layers 1 "

    lora_args = ""
    if args.train_mode == "full":
        # NVMe-streamed optimizer state (GPU-stepped, one bucket resident) + tms
        # disk backup for the paused training actor; fixed micro-batches (dynamic
        # token packing exposes a PP-p2p x EP-a2a NCCL launch-order race on
        # varlen shapes).
        optimizer_args += (
            f"--optimizer-state-nvme-dir {args.optimizer_nvme_dir} "
            "--optimizer-state-nvme-chunk-mb 256 "
            "--offload-train-target disk "
            f"--offload-train-disk-dir {args.train_offload_disk_dir} "
        )
        perf_args += "--micro-batch-size 1 "
        sglang_args = (
            "--rollout-num-gpus-per-engine 16 "
            "--sglang-mem-fraction-static 0.6 "
            "--sglang-max-running-requests 64 "
            "--sglang-max-total-tokens 327680 "
        )
    else:
        lora_args = (
            f"--lora-rank {args.lora_rank} "
            f"--lora-alpha {args.lora_alpha} "
            "--target-modules all-linear "
            "--sglang-lora-backend triton "
            "--sglang-lora-use-virtual-experts "
            "--sglang-max-loras-per-batch 1 "
            f"--sglang-max-lora-rank {args.lora_rank} "
        )
        if args.lora_adapter_path is not None:
            lora_args += f"--lora-adapter-path {args.lora_adapter_path} "
        perf_args += "--use-dynamic-batch-size " "--max-tokens-per-gpu 4096 "
        sglang_args = (
            "--rollout-num-gpus-per-engine 16 "
            "--sglang-ep-size 16 "
            "--no-offload-rollout --no-offload-train "
            "--sglang-mem-fraction-static 0.65 "
            "--sglang-max-running-requests 32 "
            "--sglang-max-total-tokens 320000 "
            "--sglang-cuda-graph-max-bs 64 "
            "--sglang-max-mamba-cache-size 256 "
        )

    sglang_args += (
        "--sglang-attention-backend fa4 "
        "--sglang-moe-runner-backend triton "
        "--sglang-mamba-scheduler-strategy extra_buffer "
        "--sglang-enable-multimodal "
        f"--sglang-context-length {args.sglang_context_length} "
        "--sglang-disable-custom-all-reduce "
    )

    inkling_args = "--inkling-attn-backend flex " "--inkling-freeze-global-scale all "
    if args.is_mm:
        inkling_args += "--inkling-mm-towers "

    misc_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--no-bias-dropout-fusion "
        "--distributed-timeout-minutes 30 "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    extra_env_vars = {
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        "SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM": "false",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "MILES_SGLANG_DUMMY_LOAD": "0",
        "SGLANG_SERVER_ENGINE_ROLLOUT_RETURN_LOGPROB": "1",
        "RAY_memory_monitor_refresh_ms": "0",
        "NCCL_MNNVL_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_RAS_ENABLE": "0",
    }

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{perf_args} "
        f"{inkling_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="inkling-975b",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training. Assumes HF checkpoint / torch_dist are already staged."""
    _train(args)


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(path_src=args.torch_dist, path_dst=args.torch_dist_local)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    """Copy the shared torch_dist checkpoint to node-local NVMe (torch_dist_local)."""
    _prepare_cp(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    if args.torch_dist_local != args.torch_dist:
        _prepare_cp(args)
    else:
        print(f"[full_train] Skipping rsync: torch_dist_local == torch_dist ({args.torch_dist})")
    _train(args)


if __name__ == "__main__":
    app()
