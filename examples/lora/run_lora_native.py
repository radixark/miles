"""GRPO LoRA training through the native (raw-mode) path for MoE registries.

Companion to ``examples/lora/run_qwen3_lora_native.py``, which covers the dense GQA
reference recipe. This one drives the shipped MoE registries whose attention is
either multi-latent (GLM / DeepSeek / Kimi) or a gated GQA hybrid (Qwen3.5), i.e.
exactly the layouts ``miles_plugins.lora`` grew beyond the plain fused-qkv case
for.

Coverage is attention-only on purpose. Routed-expert adapters need a serving-side
layout contract of their own, so ``--target-modules`` lists just the attention
projections and the trainer and the engine agree on which modules carry an adapter:

* MLA registries: ``q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj``
  (SGLang stacks the two down-projections into ``fused_qkv_a_proj_with_mqa``)
* Qwen3.5: ``q_proj,k_proj,v_proj,o_proj``, where ``q_proj`` carries the output-gate
  slice. Its linear-attention (GDN) layers have no fused qkv and get no adapter.

A raw-mode Qwen3.5 backward divergence (grad_norm 1e7-1e10 with recompute, NaN without)
was once on record here; it no longer reproduces — 20-rollout runs of both Qwen3.5-35B-A3B
and Qwen3.5-9B hold train/rollout logprob_abs_diff at ~1e-2 with grad_norm ~1e-2 throughout.

Every registry here needs a raw-mode torch-dist checkpoint of the BF16 base; the
``prepare`` command builds one, dequantizing first when the published checkpoint is
FP8 or INT4.

Two defaults are deliberate rather than conventional. ``lr`` is 1e-4 so the adapter
becomes meaningfully nonzero within a few steps: while ``B`` is still at its zero
init the LoRA delta vanishes, and train/rollout agreement then says nothing about
whether the exported adapter was served correctly. ``rollout_batch_size`` is 32
because at 8 prompts nearly every gsm8k group comes out all-correct, GRPO's
advantage is zero, and the adapter never moves.

Usage:
  python examples/lora/run_lora_native.py prepare    --model-name GLM-4.7-Flash
  python examples/lora/run_lora_native.py train      --model-name GLM-4.7-Flash
  python examples/lora/run_lora_native.py full-train --model-name Qwen3.5-35B-A3B
"""

from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MLA_TARGET_MODULES = "q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
_GQA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"


@dataclass(frozen=True)
class _Registry:
    """What differs per checkpoint: the megatron registry, the base, the parallel shape."""

    megatron_model_type: str
    hf_repo: str
    target_modules: str
    tensor_model_parallel_size: int
    expert_model_parallel_size: int
    rollout_num_gpus_per_engine: int
    dequantize: Literal["none", "fp8", "kimi-int4"] = "none"
    extra_train_args: str = ""


_REGISTRIES = {
    "GLM-4.7-Flash": _Registry(
        megatron_model_type="glm4.7-flash",
        hf_repo="zai-org/GLM-4.7-Flash",
        target_modules=_MLA_TARGET_MODULES,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=4,
        rollout_num_gpus_per_engine=2,
        extra_train_args="--attention-backend flash ",
    ),
    "DeepSeek-V4-Flash-FP8-4layer": _Registry(
        megatron_model_type="deepseek-v4-flash-4layer",
        hf_repo="Pinaster/DeepSeek-V4-Flash-FP8-4layer",
        target_modules=_MLA_TARGET_MODULES,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=4,
        rollout_num_gpus_per_engine=2,
        dequantize="fp8",
    ),
    "Kimi-K2.5-2layer": _Registry(
        megatron_model_type="kimi-k25_2layer",
        hf_repo="CharyZeng/Kimi-K2.5-2layer",
        target_modules=_MLA_TARGET_MODULES,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=4,
        rollout_num_gpus_per_engine=2,
        dequantize="kimi-int4",
    ),
    "Qwen3.5-35B-A3B": _Registry(
        megatron_model_type="qwen3.5-35B-A3B",
        hf_repo="Qwen/Qwen3.5-35B-A3B",
        target_modules=_GQA_TARGET_MODULES,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=4,
        rollout_num_gpus_per_engine=2,
    ),
    "Qwen3.5-9B": _Registry(
        megatron_model_type="qwen3.5-9B",
        hf_repo="Qwen/Qwen3.5-9B",
        target_modules=_GQA_TARGET_MODULES,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=1,
        rollout_num_gpus_per_engine=2,
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "GLM-4.7-Flash"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    model_dir: str = "/scratch/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 0
    expert_model_parallel_size: int = 0
    rollout_num_gpus_per_engine: int = 0
    max_tokens_per_gpu: int = 8192
    recompute: bool = True

    lr: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = ""
    debug_lora_train_only: bool = False
    check_lora_weight_equal: bool = True

    num_rollout: int = 8
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 2048
    global_batch_size: int = 128

    sglang_mem_fraction_static: float = 0.5
    sglang_lora_backend: str = "triton"

    enable_wandb: bool = False
    extra_args: str = ""
    registry: _Registry = field(init=False)

    def __post_init__(self):
        assert self.model_name in _REGISTRIES, f"{self.model_name} is not a native-LoRA registry"
        self.registry = _REGISTRIES[self.model_name]
        if not self.target_modules:
            self.target_modules = self.registry.target_modules
        for name in ("tensor_model_parallel_size", "expert_model_parallel_size", "rollout_num_gpus_per_engine"):
            if getattr(self, name) == 0:
                setattr(self, name, getattr(self.registry, name))

    @property
    def megatron_model_type(self) -> str:
        return self.registry.megatron_model_type

    @property
    def hf_checkpoint(self) -> str:
        """The BF16 base: the download itself, or the dequantized copy beside it."""
        if self.registry.dequantize == "none":
            return f"{self.model_dir}/{self.model_name}"
        return f"{self.model_dir}/{self.model_name}-bf16"

    @property
    def torch_dist(self) -> str:
        return f"{self.model_dir}/{self.model_name}_torch_dist"


def _download(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command_cpu(f"hf download {args.registry.hf_repo} --local-dir {args.model_dir}/{args.model_name}")
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _dequantize(args: ScriptArgs):
    src = f"{args.model_dir}/{args.model_name}"
    match args.registry.dequantize:
        case "fp8":
            U.fp8_cast_bf16(path_src=src, path_dst=args.hf_checkpoint)
        case "kimi-int4":
            U.exec_command_gpu(
                f"python {U.repo_base_dir}/tools/convert_kimi_int4_to_bf16.py "
                f"--model-dir {src} --output-dir {args.hf_checkpoint}"
            )


def _convert(args: ScriptArgs):
    _preflight(args)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=min(args.num_gpus_per_node, 4),
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args=(
            "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 --context-parallel-size 1 "
        ),
    )


def _prepare(args: ScriptArgs):
    _download(args)
    _dequantize(args)
    _convert(args)


def _preflight(args: ScriptArgs) -> None:
    """Audit registry/mbridge/model-args coverage before any GPU work."""
    try:
        from miles_plugins.lora import preflight_native_lora
    except ImportError:  # older plugin without the helper
        return
    report = preflight_native_lora(args.hf_checkpoint, args.megatron_model_type, strict=True)
    print(report.render(), flush=True)


def _train(args: ScriptArgs):
    _preflight(args)
    print(
        f"[run] native LoRA: {args.model_name} (megatron_model_type={args.megatron_model_type}), "
        f"TP{args.tensor_model_parallel_size} EP{args.expert_model_parallel_size}, "
        f"targets={args.target_modules}"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --load {args.torch_dist} "
        "--megatron-to-hf-mode raw --no-load-optim --no-load-rng --finetune "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        f'--lora-dropout {args.lora_dropout} --target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
    )
    if args.debug_lora_train_only:
        lora_args += "--debug-lora-train-only "
    if args.check_lora_weight_equal:
        lora_args += "--check-lora-weight-equal "

    rollout_args = (
        "--label-key label --apply-chat-template --rollout-shuffle --rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )
    match args.task:
        case "gsm8k":
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} --sequence-parallel "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} --expert-tensor-parallel-size 1 "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )
    if args.recompute:
        perf_args += "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-dtype bfloat16 --sglang-decode-log-interval 1000 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--update-weight-buffer-size 536870912 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} {wandb_args} "
        f"{perf_args} {sglang_args} {misc_args} {args.registry.extra_train_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the checkpoint + dataset, dequantize if needed, build the torch-dist base."""
    _prepare(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training through the native path (assumes prepare already ran)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Prepare, then run GRPO LoRA training through the native path."""
    _prepare(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
