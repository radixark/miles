"""GLM-5.2 GRPO LoRA training through the native (raw-mode) path.

Native-path companion to ``scripts/run_glm5_2_744b_a40b_lora.py`` (bridge). The
model is built by miles' own glm5 provider (``--spec miles_plugins.models.glm5``)
and ``miles_plugins.lora`` attaches the adapters (model_type ``glm_moe_dsa`` ->
MLA arch spec). Coverage is attention-only, matching the other MLA registries in
``examples/lora/run_lora_native.py``: routed-expert adapters stay out of scope for the
native provider.

DSA notes carried over from the bridge script: the kernel backend dictates the
query layout (tilelang => thd, megatron => bshd), both forbid
--use-dynamic-batch-size, and serving needs the nsa attention backend with
flashmla_sparse kernels.

Usage:
  python examples/lora/run_glm5_2_744b_a40b_lora_native.py prepare    --model-name GLM-5.2_5layer
  python examples/lora/run_glm5_2_744b_a40b_lora_native.py train      --model-name GLM-5.2_5layer --num-rollout 20
  python examples/lora/run_glm5_2_744b_a40b_lora_native.py full-train --model-name GLM-5.2_5layer
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# attention-only, same set as the MLA registries in run_lora_native.py
_MLA_TARGET_MODULES = "q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"

_HF_REPO = {
    "GLM-5.2_5layer": "Pinaster/GLM-5.2_5layer",
}

_MEGATRON_MODEL_TYPE = {
    "GLM-5.2_5layer": "glm5.2-744B-A40B_5layer_lora",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["GLM-5.2_5layer"] = "GLM-5.2_5layer"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/scratch/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # the matching --qkv-format is derived from this (see _get_parallel_config)
    dsa_attention_backend: Literal["megatron", "tilelang"] = "tilelang"

    # R3 rollout routing replay (arxiv 2510.11370)
    use_r3: bool = True

    num_gpus_per_node: int = 4

    lr: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _MLA_TARGET_MODULES
    check_lora_weight_equal: bool = True

    num_rollout: int = 8
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 0  # 0 => per-task default (gsm8k 512, dapo-math 4096)
    seq_window: int = 0
    global_batch_size: int = 128

    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.5
    # sglang's own default (csgmv) crashes the DSA LoRA rollout under dp-attention
    sglang_lora_backend: str = "triton"

    enable_wandb: bool = False
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

    @property
    def torch_dist(self) -> str:
        return f"{self.model_dir}/{self.model_name}_torch_dist"


def _get_parallel_config(args: ScriptArgs) -> str:
    """Single-node MoE layout: TP = EP = num_gpus_per_node, DP1 (mirrors the bridge script).

    Both DSA kernel backends forbid --use-dynamic-batch-size, hence
    --micro-batch-size 1: megatron needs bshd, tilelang needs thd.
    """
    ngpu = args.num_gpus_per_node
    qkv_format = "thd" if args.dsa_attention_backend == "tilelang" else "bshd"
    return (
        f"--tensor-model-parallel-size {ngpu} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {ngpu} --expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} --micro-batch-size 1 "
    )


def _download(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command_cpu(f"hf download {_HF_REPO[args.model_name]} --local-dir {args.model_dir}/{args.model_name}")
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _preflight(args: ScriptArgs) -> None:
    """Audit registry/mbridge/model-args coverage before any GPU work."""
    try:
        from miles_plugins.lora import preflight_native_lora
    except ImportError:  # older plugin without the helper
        return
    report = preflight_native_lora(args.hf_checkpoint, args.megatron_model_type, strict=True)
    print(report.render(), flush=True)


def _convert(args: ScriptArgs):
    _preflight(args)
    # Single-rank conversion: the convert tool pipeline-shards across its ranks,
    # and any PP split of the 5-layer toy starts a stage on a DSA skip layer
    # (cross-layer top-k sharing cannot cross PP boundaries -> provider assert).
    # The toy is ~50 GB of bf16 weights, well within one H200.
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=1,
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
    _convert(args)


def _train(args: ScriptArgs):
    _preflight(args)
    print(
        f"[run] native LoRA: {args.model_name} (megatron_model_type={args.megatron_model_type}), "
        f"dsa-backend={args.dsa_attention_backend}, r3={args.use_r3}, {args.num_gpus_per_node} GPUs, "
        f"targets={args.target_modules}"
    )
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --load {args.torch_dist} "
        "--megatron-to-hf-mode raw --no-load-optim --no-load-rng --finetune "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        f'--lora-dropout {args.lora_dropout} --target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
    )
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

    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    perf_args = _get_parallel_config(args)

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-cuda-graph-max-bs 64 --sglang-moe-runner-backend triton "
        "--sglang-disable-shared-experts-fusion "
        "--sglang-reasoning-parser glm45 --sglang-tool-call-parser glm47 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
    )

    save_args = f"--save-interval 1 --save {load_save_path} "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss "
        f"--use-miles-router --actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --colocate "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    seq_args = (
        f"--seq-length {args.seq_window} --rollout-max-context-len {args.seq_window} " if args.seq_window > 0 else ""
    )

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {seq_args} {optimizer_args} {grpo_args} {r3_args} "
        f"{wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            # GLM-5 DSA indexer uses interleaved RoPE; a mismatch garbles long sequences
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "SGLANG_NSA_FORCE_MLA": "1",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the checkpoint + dataset and build the raw-mode torch-dist base."""
    _prepare(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training through the native path (assumes prepare already ran)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Prepare, then train."""
    _prepare(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
