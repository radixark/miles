"""
GLM-5.1 744B-A40B GRPO LoRA training script (Megatron-Bridge / bridge mode).

GLM-5.1 is MoE + MLA + DSA. Unlike GLM-5.2 it has NO DSA cross-layer index sharing (every
layer carries its own indexer); the registries are identical to the GLM-5.2 ones except
``--rotary-base`` 1e6 (5.2 uses 8e6). LoRA trains through the bridge path
(``--megatron-to-hf-mode bridge``); the registry ``.sh`` may still list ``--spec``, which
is inert under bridge LoRA. GLM-5.2 lives in ``scripts/run_glm5_2_744b_a40b_lora.py``.

DSA kernel backend (``--dsa-attention-backend``; orthogonal to model version and to LoRA):
  * ``tilelang`` (default): vendored fused TileLang kernels, thd (packed) layout, needs
    the optional ``tilelang`` dep; matches slime's rollout kernels for rollout<->train
    numerical parity. Training/forward-only -- the rollout is always served by sglang.
  * ``megatron``: portable unfused megatron-core DSA kernels, bshd layout,
    no extra deps.
The matching ``--qkv-format`` is selected automatically (see ``_get_parallel_config``).

``--target-modules`` excludes the 3 DSA indexer modules (wq_b/wk/weights_proj) by default:
on tilelang the indexer adapter gets no gradient at all; on megatron it
would only get a tiny aux-loss gradient (~1e-5).

Supported model variants (HF checkpoint must be the native config,
model_type=glm_moe_dsa / GlmMoeDsaForCausalLM):
  GLM-5.1          full 744B model (zai-org/GLM-5.1)
  GLM-5.1-6layer   6-layer GLM-5.1 prune (jybsuper/GLM-5.1-6layer; 3 dense + 3 MoE)

Usage:
  python scripts/run_glm5_1_744b_a40b_lora.py prepare    --model-name GLM-5.1-6layer
  python scripts/run_glm5_1_744b_a40b_lora.py full-train --model-name GLM-5.1-6layer --num-gpus-per-node 4
  python scripts/run_glm5_1_744b_a40b_lora.py full-train --model-name GLM-5.1-6layer \\
      --dsa-attention-backend megatron --num-gpus-per-node 4
  python scripts/run_glm5_1_744b_a40b_lora.py prepare --model-name GLM-5.1-6layer --task dapo-math
  python scripts/run_glm5_1_744b_a40b_lora.py train   --model-name GLM-5.1-6layer --task dapo-math \\
      --rollout-max-response-len 4096 --num-gpus-per-node 4
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = {
    "GLM-5.1": "zai-org/GLM-5.1",
    "GLM-5.1-6layer": "jybsuper/GLM-5.1-6layer",
}

_MEGATRON_MODEL_TYPE = {
    "GLM-5.1": "glm5.1-744B-A40B_lora",
    "GLM-5.1-6layer": "glm5.1-744B-A40B_6layer_lora",
}

# Standard attn + MLA + MLP/MoE, EXCLUDING the DSA indexer (wq_b/wk/weights_proj).
_DEFAULT_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal[
        "GLM-5.1",
        "GLM-5.1-6layer",
    ] = "GLM-5.1-6layer"
    # dapo-math needs a larger --rollout-max-response-len; >2048 total seq makes the DSA indexer sparse
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # the matching --qkv-format is derived from this (see _get_parallel_config)
    dsa_attention_backend: Literal["megatron", "tilelang"] = "tilelang"

    # R3 rollout routing replay (arxiv 2510.11370)
    use_r3: bool = True

    # performance
    num_gpus_per_node: int = 4

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    # required for true on-policy under colocate (OFF -> KL ~1.0 vs ~1e-4); opt out only
    # when host RAM cannot take the ~372 GB/node mirror on the full model
    lora_base_cpu_backup: bool = True
    # MoE-expert LoRA layout: shared-outer when True, per-expert when False
    experts_shared_outer_loras: bool = True

    # rollout
    num_rollout: int = 1
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 0  # 0 => per-task default (gsm8k 512, dapo-math 4096)
    # emitted as --seq-length + --rollout-max-context-len when > 0
    seq_window: int = 0
    global_batch_size: int = 16

    # OFF by default: a model scoring 0 on every sample (e.g. the toys) would resample forever
    dapo_dynamic_sampling: bool = False
    over_sampling_batch_size: int = 32  # used only when dapo_dynamic_sampling; should exceed rollout_batch_size

    # rollout engine
    rollout_num_gpus_per_engine: int = 2  # rollout tp=2
    sglang_mem_fraction_static: float = 0.5
    # sglang's own default (csgmv) crashes the DSA MoE-LoRA rollout under dp-attention
    sglang_lora_backend: str = "triton"
    # serve from a pre-converted _fp8 ckpt (fits engine=8 / 1 node); train stays bf16
    fp8_rollout: bool = False

    enable_wandb: bool = True
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


def _get_parallel_config(args: ScriptArgs) -> str:
    """Single-node MoE layout: TP = EP = num_gpus_per_node, DP1 (mirrors run_glm5_744b_a40b).

    The DSA kernel backend dictates the query layout; both forbid --use-dynamic-batch-size,
    hence --micro-batch-size 1: megatron needs bshd (the unfused megatron-core
    DSA core-attention takes a 4D query), tilelang needs thd (the fused kernels index by
    cu_seqlens).
    """
    ngpu = args.num_gpus_per_node
    qkv_format = "thd" if args.dsa_attention_backend == "tilelang" else "bshd"
    return (
        f"--tensor-model-parallel-size {ngpu} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {ngpu} --expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} --micro-batch-size 1 "
    )


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.data_dir} {args.model_dir}")
    repo = _HF_REPO.get(args.model_name)
    if repo is not None:
        U.exec_command(f"hf download {repo} --local-dir {args.model_dir}/{args.model_name}")
    _download_dataset(args)


def _train(args: ScriptArgs):
    print(
        f"[run] GLM-5.1 LoRA: model={args.model_name} (megatron_model_type={args.megatron_model_type}), dsa-backend={args.dsa_attention_backend}, r3={args.use_r3}, {args.num_gpus_per_node} GPUs, rollout tp={args.rollout_num_gpus_per_engine}"
    )
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )

    # the full rollout config applies to the toys too (same glm_moe_dsa serving path)
    _is_full = True
    _tm = args.target_modules
    # KEEP_MOE_LORA=0 drops the expert projections (attention-only LoRA)
    _keep_moe_lora = os.environ.get("KEEP_MOE_LORA", "1") != "0"
    if _is_full and not _keep_moe_lora:
        _tm = ",".join(m for m in _tm.split(",") if m.strip() not in ("gate_proj", "up_proj", "down_proj"))
    # the MOE_LORA_LAYERS subset feature is disabled; warn so it is not silently ignored
    _moe_lora_layers = os.environ.get("MOE_LORA_LAYERS", "").strip()
    if _moe_lora_layers:
        print(
            f"[run_glm5_1_744b_a40b_lora] WARNING: MOE_LORA_LAYERS={_moe_lora_layers} is SET but the subset-rewrite "
            "feature is DISABLED (commented out for debugging) -> MoE-expert LoRA stays on ALL layers."
        )
    lora_args = f'--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} --target-modules "{_tm}" '
    if _keep_moe_lora and args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if _is_full:
        lora_args += "--no-gradient-accumulation-fusion "
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )
    match args.task:
        case "gsm8k":  # zhuzilin/gsm8k ships {messages, label} parquet
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":  # zhuzilin/dapo-math-17k ships {prompt, label} jsonl (prompt = chat messages)
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "
    if args.dapo_dynamic_sampling:
        rollout_args += (
            f"--over-sampling-batch-size {args.over_sampling_batch_size} "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    grpo_args = "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    # routing replay only: --use-rollout-indexer-replay is debug-only and its
    # ~78-128 GB/rank host buffer OOMs the colocate pod
    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""

    optimizer_args = (
        "--optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    # the three CPU-Adam flags go together; OPTIMIZER_CPU_OFFLOAD=0 to disable
    if os.environ.get("OPTIMIZER_CPU_OFFLOAD", "1") != "0":
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    perf_args = _get_parallel_config(args)

    if _is_full:
        # mirrors run_glm5_744b_a40b.py; bf16 ~1488GB needs >=~22 GPUs/engine while fp8
        # fits engine=min(8, ngpu) on one node
        _eng = min(8, args.num_gpus_per_node) if args.fp8_rollout else args.rollout_num_gpus_per_engine
        _decode = "flashmla_kv" if args.fp8_rollout else "flashmla_sparse"
        _cg = 256 if args.fp8_rollout else 64
        _kv = "--sglang-kv-cache-dtype fp8_e4m3 " if args.fp8_rollout else ""
        sglang_args = (
            f"--rollout-num-gpus-per-engine {_eng} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
            f"--sglang-enable-dp-attention --sglang-ep-size {_eng} --sglang-dp-size {_eng} "
            "--sglang-moe-dense-tp-size 1 --sglang-enable-dp-lm-head "
            f"--sglang-attention-backend nsa --sglang-nsa-decode-backend {_decode} "
            f"--sglang-nsa-prefill-backend flashmla_sparse --sglang-page-size 64 {_kv}"
            f"--sglang-cuda-graph-max-bs {_cg} --sglang-max-running-requests 512 "
            f"--sglang-chunked-prefill-size {2048 * _eng} --sglang-watchdog-timeout 3600 "
            "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
            # required: without it sglang miscounts the gate_up slices -> engine-init crash
            f"--sglang-max-lora-rank {args.lora_rank} "
            f"--sglang-lora-backend {args.sglang_lora_backend} "
        )
    else:
        sglang_args = f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} --sglang-cuda-graph-max-bs 64 --sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion --sglang-lora-backend {args.sglang_lora_backend} --sglang-reasoning-parser glm45 --sglang-tool-call-parser glm47 "

    save_args = f"--save-interval 1 --save {load_save_path} "

    misc_args = f"--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss --use-miles-router --actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --num-gpus-per-node {args.num_gpus_per_node} --colocate "

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    seq_args = (
        f"--seq-length {args.seq_window} --rollout-max-context-len {args.seq_window} " if args.seq_window > 0 else ""
    )

    train_args = f"{ckpt_args} {lora_args} {rollout_args} {seq_args} {optimizer_args} {grpo_args} {r3_args} {wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "

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
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True breaks torch_memory_saver
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the model checkpoint (for a known HF repo) and the task dataset (gsm8k or dapo-math). Run once per node before training."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training (assumes the dataset is already prepared)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download the model checkpoint + dataset, then run GRPO LoRA training."""
    _prepare_download(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
