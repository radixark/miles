"""Qwen3.8-2.4T-A95B RL training launcher (NVFP4 rollout + BF16 trainer).

The rollout engine serves the native ModelOpt NVFP4 experts-only checkpoint
(--hf-checkpoint); the Megatron trainer runs BF16 from a torch_dist produced
out of the dequantized checkpoint (--ref-load). Weight updates re-quantize
expert weights to NVFP4 at the update boundary; everything else ships BF16.

Checkpoints are looked up under --model-dir as
Qwen3.8-2.4T-A95B-{NVFP4,bf16,bf16_torch_dist}_{model_variant}.

--lora runs the native (raw-mode) LoRA path against the same trainer and
torch_dist base as the full-weight run. Targets follow the qwen3.5 hybrid
recipe: attention projections only, so the rollout MoE stays on
flashinfer_trtllm unchanged.
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

# Matches the rollout checkpoint's ModelOpt export recipe: plain per-block max
# scaling, 1D group-16 E2M1 weights + E4M3 block scales, no 4-over-6 remap.
NVFP4_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    # sglang-kernel 0.4.5: the 0.4.6.post1 wheel is ABI-incompatible with the
    # image's torch 2.11+cu130, and the API surface used here matches.
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    # Match the trainer's --moe-router-dtype fp32. Rounding router logits to
    # bf16 costs ~0.25 absolute, enough to reorder near-tied experts.
    "SGLANG_MOE_ROUTER_FP32": "1",
}

EXTRA_HIGH_PRECISION_LAYERS_MEGATRON = (
    ".shared_experts.linear_fc1",
    ".shared_experts.linear_fc2",
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_variant: Literal["4layer", "full"] = "4layer"
    model_name: str = "Qwen3.8-2.4T-A95B"
    num_gpus_per_node: int = 8
    hardware: Literal["B300", "GB300"] = "B300"
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # parallelism knobs
    tp: int = 1
    ep: int = 8
    cp: int = 1
    pp: int = 1
    etp: int = 1

    # training knobs
    num_rollout: int = 2
    max_tokens_per_gpu: int = 2048
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 2
    global_batch_size: int = 8
    rollout_max_response_len: int = 64

    # LoRA knobs (HF projection leaves, full-attention layers only)
    lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    # rollout knobs
    sglang_ep_size: int | None = None
    rollout_max_concurrency: int = 16
    recompute: bool = False
    skip_prepare: bool = False
    check_weight_update: bool = True

    @property
    def nvfp4_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}-NVFP4_{self.model_variant}"

    @property
    def bf16_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}-bf16_{self.model_variant}"

    @property
    def torch_dist_checkpoint(self) -> str:
        return f"{self.bf16_checkpoint}_torch_dist"

    @property
    def megatron_model_type(self) -> str:
        return f"qwen3.8-2.4T-A95B_{self.model_variant}"


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.data_dir}")
    U.exec_command_cpu(
        f"test -e {args.data_dir}/dapo-math-17k || "
        f"hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir {args.data_dir}/dapo-math-17k"
    )
    U.exec_command_cpu(f"test -d {args.nvfp4_checkpoint}")
    U.exec_command_cpu(f"test -d {args.bf16_checkpoint}")

    U.convert_checkpoint(
        model_name=f"{args.model_name}-bf16_{args.model_variant}",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.bf16_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.nvfp4_checkpoint} "
        f"--ref-load {args.torch_dist_checkpoint} "
        "--extra-high-precision-layers-megatron " + " ".join(EXTRA_HIGH_PRECISION_LAYERS_MEGATRON) + " "
    )

    lora_args = ""
    if args.lora:
        lora_args = (
            f"--lora-rank {args.lora_rank} "
            f"--lora-alpha {args.lora_alpha} "
            f"--lora-dropout {args.lora_dropout} "
            f'--target-modules "{args.target_modules}" '
            "--no-gradient-accumulation-fusion "
            # Base sleeps into the sglang-side CPU mirror instead of being
            # re-shipped and re-quantized on every update (~200 GB/rank staging).
            "--lora-base-cpu-backup "
        )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    if args.mode == "debug_minimal":
        rollout_args += "--rm-type deterministic_random "
    else:
        rollout_args += "--rm-type deepscaler "

    check_args = ""
    if args.check_weight_update:
        # allow-quant-error: the trainer's requantization cannot reproduce
        # ModelOpt's export bytes bitwise, so experts compare in dequant space.
        check_args = "--check-weight-update-equal " "--check-weight-update-allow-quant-error "
        if args.lora:
            check_args += "--check-lora-weight-equal "

    recompute = (
        ("--recompute-granularity full " "--recompute-method uniform " "--recompute-num-layers 1 ")
        if args.recompute
        else ""
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pp} "
        f"--context-parallel-size {args.cp} "
        f"--expert-model-parallel-size {args.ep} "
        f"--expert-tensor-parallel-size {args.etp} "
        f"{recompute}"
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # no --use-kl-loss: it loads a full ref checkpoint and a second pinned
        # backup (~75 GB/rank) that a zero kl coefficient multiplies away.
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {'1e-4' if args.lora else '1e-6'} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_ep = args.sglang_ep_size if args.sglang_ep_size is not None else 1
    # The text-only arch is absent from sglang's auto-override registries, so
    # the backends and the mamba cache strategy must be passed explicitly.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        # Colocate offload releases the weight region, and never-pushed params
        # (kv-cache scales, NVFP4 input scales) must survive it. --lora-base-cpu-backup
        # is the restore source; the full sglang-side backup needs ~3 TB host RAM,
        # which only B300 has.
        + ("" if args.hardware == "GB300" else "--sglang-enable-weights-cpu-backup ")
        + "--sglang-mem-fraction-static 0.7 "
        f"--sglang-ep-size {sglang_ep} "
        "--sglang-attention-backend triton "
        "--sglang-moe-runner-backend flashinfer_trtllm "
        "--sglang-mamba-radix-cache-strategy extra_buffer "
        f"--sglang-max-running-requests {args.rollout_max_concurrency} "
        f"--sglang-max-mamba-cache-size {5 * args.rollout_max_concurrency} "
        "--sglang-context-length 8192 "
        "--sglang-cuda-graph-bs-decode 1 2 4 8 16 "
        "--sglang-cuda-graph-backend-prefill disabled "
    )
    if args.lora:
        sglang_args += (
            f"--sglang-max-lora-rank {args.lora_rank} " "--sglang-lora-backend triton " "--sglang-lora-strict-loading "
        )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{check_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=dict(NVFP4_ENV),
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
