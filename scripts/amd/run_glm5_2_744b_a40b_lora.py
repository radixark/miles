"""GLM-5.2 744B-A40B GRPO LoRA training script for AMD (MI350X / MI355X).

ROCm counterpart of scripts/run_glm5_2_744b_a40b_lora.py. GLM-5.2 is MoE + MLA + DSA with
cross-layer index sharing; LoRA trains through Megatron-Bridge (``--megatron-to-hf-mode
bridge``) and the rollout is served by SGLang.

Differences from the CUDA recipe:
  - SGLang NSA prefill/decode run the TileLang kernels; flashmla has no HIP build.
  - --dsa-attention-backend megatron, not tilelang. The image's Megatron-Bridge builds
    the GLM-5 MLA with fuse_input_layernorm=False, so linear_kv_up_proj is a plain
    TEColumnParallelLinear, while its tilelang_mla absorb reads layer_norm_weight off
    that module.
  - No dp-attention. A DP rank with no requests runs an IDLE forward, and SGLang's
    MoE-LoRA path rejects that mode (get_batch_token_counts), so the first idling
    step kills the engine.
  - No --sglang-moe-dense-tp-size 1. It leaves the dense MLP unsharded while the
    LoRA buffers are still sized input_dim / tp_size, so down_proj trips
    `assert x.shape[-1] == K` during CUDA-graph capture.
  - The CPU-Adam trio is dropped: it exists to fit an 80 GB card, and it changes
    optimizer numerics.

``--target-modules`` excludes the 3 DSA indexer modules (wq_b/wk/weights_proj): on the
megatron backend the indexer adapter only gets a tiny aux-loss gradient (~1e-5), and on
tilelang it gets none at all.

Model variants (the HF checkpoint must be the native glm_moe_dsa config):
  GLM-5.2         full 744B model (zai-org/GLM-5.2)
  GLM-5.2_5layer  5-layer prune (Pinaster/GLM-5.2_5layer; 3 dense + 2 MoE)

Examples:
  python scripts/amd/run_glm5_2_744b_a40b_lora.py prepare    --model-name GLM-5.2_5layer
  python scripts/amd/run_glm5_2_744b_a40b_lora.py full-train --model-name GLM-5.2_5layer
  python scripts/amd/run_glm5_2_744b_a40b_lora.py train      --model-name GLM-5.2_5layer \
      --task dapo-math --rollout-max-response-len 4096
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = {
    "GLM-5.2": "zai-org/GLM-5.2",
    "GLM-5.2_5layer": "Pinaster/GLM-5.2_5layer",
}

_MEGATRON_MODEL_TYPE = {
    "GLM-5.2": "glm5.2-744B-A40B_lora",
    "GLM-5.2_5layer": "glm5.2-744B-A40B_5layer_lora",
}

# Standard attn + MLA + MLP/MoE, excluding the DSA indexer (wq_b/wk/weights_proj).
_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["GLM-5.2", "GLM-5.2_5layer"] = "GLM-5.2_5layer"
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None
    # dapo-math needs a larger --rollout-max-response-len; >2048 total seq makes the DSA indexer sparse
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # the matching --qkv-format is derived from this (see _get_parallel_config)
    dsa_attention_backend: Literal["megatron", "tilelang"] = "megatron"
    # R3 rollout routing replay (arxiv 2510.11370)
    use_r3: bool = True

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _TARGET_MODULES
    # required for true on-policy under colocate (OFF -> KL ~1.0 vs ~1e-4)
    lora_base_cpu_backup: bool = True
    # MoE-expert LoRA layout: shared-outer when True, per-expert when False
    experts_shared_outer_loras: bool = True

    num_rollout: int = 20
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 0  # 0 => per-task default (gsm8k 512, dapo-math 4096)
    # emitted as --seq-length + --rollout-max-context-len when > 0
    seq_window: int = 0
    global_batch_size: int = 64
    lr: float = 1e-5

    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.5

    # 0 => no --save. Saving works with these targets (the adapter and the HF PEFT
    # export are both written); it is off so a plain run leaves nothing behind.
    save_interval: int = 0
    enable_wandb: bool = True
    wandb_team: str | None = None
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = 4096 if self.task == "dapo-math" else 512
        if self.seq_window == 0 and self.task == "dapo-math":
            self.seq_window = 8192

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _set_rocm_environment() -> None:
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _resolve_num_gpus(args: ScriptArgs) -> tuple[str, int]:
    hardware = U.resolve_hardware(args)
    return hardware, args.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[hardware]


def _get_parallel_config(num_gpus: int, args: ScriptArgs) -> str:
    """Single-node MoE layout: TP = EP = num_gpus, DP1 (mirrors the CUDA recipe).

    The DSA kernel backend dictates the query layout; both forbid --use-dynamic-batch-size,
    hence --micro-batch-size 1. megatron needs bshd (the unfused megatron-core DSA
    core-attention takes a 4D query), tilelang needs thd (the fused kernels index by
    cu_seqlens).
    """
    qkv_format = "thd" if args.dsa_attention_backend == "tilelang" else "bshd"
    return (
        f"--tensor-model-parallel-size {num_gpus} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {num_gpus} --expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} --micro-batch-size 1 "
    )


def _download_inputs(args: ScriptArgs) -> None:
    U.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command_cpu(f"hf download {_HF_REPO[args.model_name]} --local-dir {args.model_dir}/{args.model_name}")
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _get_wandb_args(args: ScriptArgs) -> str:
    if not args.enable_wandb:
        return ""
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args and args.wandb_team:
        wandb_args += f"--wandb-team {shlex.quote(args.wandb_team)} "
    return wandb_args


def _execute(args: ScriptArgs) -> None:
    _set_rocm_environment()
    hardware, num_gpus = _resolve_num_gpus(args)
    print(
        f"[run] GLM-5.2 LoRA on {hardware}: model={args.model_name}, "
        f"dsa-backend={args.dsa_attention_backend}, r3={args.use_r3}, {num_gpus} GPUs, "
        f"rollout tp={args.rollout_num_gpus_per_engine}"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" --no-gradient-accumulation-fusion '
    )
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    rollout_args = (
        "--label-key label --apply-chat-template --rollout-shuffle --rm-type math "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} --rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )
    match args.task:
        case "gsm8k":  # zhuzilin/gsm8k ships {messages, label} parquet
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":  # zhuzilin/dapo-math-17k ships {prompt, label} jsonl
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "

    seq_args = (
        f"--seq-length {args.seq_window} --rollout-max-context-len {args.seq_window} " if args.seq_window > 0 else ""
    )

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )

    # routing replay only: --use-rollout-indexer-replay is debug-only and its
    # ~78-128 GB/rank host buffer OOMs the colocate pod
    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""

    perf_args = _get_parallel_config(num_gpus, args)

    engine_gpus = args.rollout_num_gpus_per_engine
    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-ep-size {engine_gpus} "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-decode-backend tilelang --sglang-nsa-prefill-backend tilelang "
        "--sglang-kv-cache-dtype bfloat16 "
        "--sglang-page-size 64 --sglang-cuda-graph-max-bs 64 --sglang-max-running-requests 512 "
        f"--sglang-chunked-prefill-size {2048 * engine_gpus} --sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
        # required: without it sglang miscounts the gate_up slices -> engine-init crash
        f"--sglang-max-lora-rank {args.lora_rank} --sglang-lora-backend triton "
    )

    save_args = (
        f"--save-interval {args.save_interval} --save {args.output_dir}/{args.run_id}/checkpoints "
        if args.save_interval > 0
        else ""
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss "
        "--moe-token-dispatcher-type alltoall "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {num_gpus} "
        f"--num-gpus-per-node {num_gpus} --colocate "
    )

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {seq_args} {optimizer_args} {grpo_args} {r3_args} "
        f"{_get_wandb_args(args)} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            # GLM-5 DSA indexer uses interleaved RoPE; a mismatch garbles long sequences
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "SGLANG_NSA_FORCE_MLA": "1",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the model checkpoint and the task dataset (gsm8k or dapo-math)."""
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
