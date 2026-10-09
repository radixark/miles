"""GLM-5.2 744B-A40B GRPO LoRA RL on 4 x 8 AMD MI355X.

gsm8k, 8 prompts x 16 samples, 256-token responses, global batch 64, LoRA rank 8 with MLP adapters
on the last 10 layers.

Examples:
  python scripts/amd/run_glm5_2_744b_a40b_lora.py prepare
  # Ray already up across the 4 nodes, MILES_SCRIPT_EXTERNAL_RAY=1:
  python scripts/amd/run_glm5_2_744b_a40b_lora.py train
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = "zai-org/GLM-5.2"
_MODEL_NAME = "GLM-5.2"
_MEGATRON_MODEL_TYPE = "glm5.2-744B-A40B_lora"
_NUM_LAYERS = 78

# The TileLang DSA decode kernel needs >= 4 of GLM-5.2's 64 heads per engine rank.
_MAX_ENGINE_GPUS = 16


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    hardware: Literal["auto", "MI355X"] = "auto"
    num_nodes: int = 4

    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    exclude_modules: str = "down_proj"
    mlp_lora_last_n: int = 10

    num_rollout: int = 40
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 16
    global_batch_size: int = 64
    seq_length: int = 1024
    rollout_max_response_len: int = 256
    # Micro-batches pad to tp_size * this; miles' default (128) pads a ~360-token sample to 1024.
    data_pad_size_multiplier: int = 16
    lr: float = 1e-5

    sglang_mem_fraction_static: float = 0.85
    # Node-local, not tmpfs: the engine already mirrors the base weights in host RAM.
    offload_train_disk_dir: str = "/root/train_offload"
    enable_wandb: bool = True
    wandb_team: str | None = None
    extra_args: str = ""


def _set_rocm_environment() -> None:
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    # Avoid execute_train's NVIDIA topology probe and disable unsupported NVLink SHARP.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _parallel_args(args: ScriptArgs, num_gpus: int) -> str:
    world_size = args.num_nodes * num_gpus
    # megatron-core's unfused DSA runs bshd and cannot recompute activations.
    return (
        f"--tensor-model-parallel-size {num_gpus} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {world_size} --expert-tensor-parallel-size 1 "
        "--qkv-format bshd --micro-batch-size 1 "
    )


def _download_inputs(args: ScriptArgs) -> None:
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    backend.exec_command_cpu(f"hf download {_HF_REPO} --local-dir {args.model_dir}/{_MODEL_NAME}")
    backend.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _get_wandb_args(args: ScriptArgs) -> str:
    if not args.enable_wandb:
        return ""
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args and args.wandb_team:
        wandb_args += f"--wandb-team {shlex.quote(args.wandb_team)} "
    return wandb_args


def _execute(args: ScriptArgs) -> None:
    _set_rocm_environment()
    hardware = U.resolve_hardware(args)
    num_gpus = U.NUM_GPUS_OF_HARDWARE[hardware]
    world_size = args.num_nodes * num_gpus
    engine_gpus = min(world_size, _MAX_ENGINE_GPUS)
    max_running_requests = args.rollout_batch_size * args.n_samples_per_prompt
    print(
        f"[run] GLM-5.2 LoRA on {hardware}: {args.num_nodes} node(s) x {num_gpus} GPUs "
        f"(TP={num_gpus} EP={world_size}), rollout tp={engine_gpus}"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{_MODEL_NAME} --megatron-to-hf-mode bridge "
        "--dsa-attention-backend megatron "
    )

    exclude_modules = args.exclude_modules.split(",")
    exclude_modules += [f"model.layers.{n}.mlp.*" for n in range(_NUM_LAYERS - args.mlp_lora_last_n)]
    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} "
        f"--target-modules {args.target_modules} --exclude-modules {shlex.quote(','.join(exclude_modules))} "
        "--no-gradient-accumulation-fusion --experts-shared-outer-loras --lora-base-cpu-backup "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages --label-key label "
        "--apply-chat-template --rollout-shuffle --rm-type math --rollout-temperature 1.0 "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--seq-length {args.seq_length} --rollout-max-context-len {args.seq_length} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--data-pad-size-multiplier {args.data_pad_size_multiplier} "
    )

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--use-rollout-routing-replay "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-ep-size {engine_gpus} "
        "--sglang-attention-backend dsa "
        # ROCm: flashmla_sparse / flashmla_kv have no HIP build.
        "--sglang-dsa-prefill-backend tilelang --sglang-dsa-decode-backend tilelang "
        "--sglang-page-size 64 "
        f"--sglang-context-length {args.seq_length} "
        f"--sglang-cuda-graph-max-bs-decode {max_running_requests} "
        f"--sglang-max-running-requests {max_running_requests} "
        "--sglang-chunked-prefill-size 8192 --sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
        f"--sglang-max-lora-rank {args.lora_rank} --sglang-lora-backend triton "
    )

    offload_args = (
        f"--offload-train-target disk --offload-train-disk-dir {args.offload_train_disk_dir} "
        "--offload-train-disk-chunk-mb 256 "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss "
        "--moe-token-dispatcher-type alltoall --colocate "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {num_gpus} "
        f"--num-gpus-per-node {num_gpus} "
        # Building the LoRA model on the 744B checkpoint outlasts miles' 10-minute default.
        "--distributed-timeout-minutes 60 "
    )

    train_args = (
        f"{ckpt_args}{lora_args}{rollout_args}{optimizer_args}{grpo_args}"
        f"{_get_wandb_args(args)}{_parallel_args(args, num_gpus)}{sglang_args}{offload_args}{misc_args}"
        f"{args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        extra_env_vars={
            # GLM-5 DSA indexer uses interleaved RoPE; a mismatch garbles long sequences.
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "SGLANG_NSA_FORCE_MLA": "1",
            # The ROCm image's NCCL_MIN_NCHANNELS=112 makes every communicator 64-112 channels wide;
            # colocate rebuilds them each rollout, which took minutes per weight update and wake-up.
            "NCCL_MIN_NCHANNELS": "16",
            "NCCL_MAX_NCHANNELS": "16",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the model checkpoint and the gsm8k dataset. Run once per node."""
    _download_inputs(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Run GRPO LoRA training using prepared inputs."""
    _execute(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download inputs and run GRPO LoRA training."""
    _download_inputs(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
