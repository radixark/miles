"""
DeepSeek V4 training script.

Supports:
  - DeepSeek-V4-Flash-FP8         Public FP8 repackage of deepseek-ai/DeepSeek-V4-Flash
                                  (sgl-project/DeepSeek-V4-Flash-FP8, 291B, 43 layers).
                                  Verified full-model profile: 4 nodes x 8 GPUs on MI355X (gfx950).
  - DeepSeek-V4-Flash-FP8-4layer  4-layer prune of the above for single-node
                                  smoke testing. **Cannot generate meaningful output -
                                  pipeline-only sanity check.**

Args:
  --fp8-recipe auto|blockwise|tensorwise: TE training recipe (default: auto).
      Auto preserves blockwise when TE supports it; on gfx942 it can select
      tensorwise E4M3 (Float8CurrentScaling). Unsupported explicit recipes fail
      before Ray submission. This does not change the rollout checkpoint format.
  --no-fp8-training: explicitly disable FP8 training; never an automatic fallback.

Usage patterns:

  1. One-shot full pipeline (download + convert + train):
       python scripts/run_deepseek_v4.py full-train \
           --model-name DeepSeek-V4-Flash-FP8-4layer \
           --num-nodes 1 --num-gpus-per-node 8

  2. Individual steps (download -> FP8->BF16 -> BF16->torch_dist -> rsync -> train):
       python scripts/run_deepseek_v4.py prepare-download --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-single   --model-name DeepSeek-V4-Flash-FP8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-spmd     --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 1 --num-gpus-per-node 8
       python scripts/run_deepseek_v4.py prepare-cp       --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py train            --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 4 --num-gpus-per-node 8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
"""

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_DEFAULT_MODEL_ORG = {
    "DeepSeek-V4-Flash-FP8": "sgl-project",
    # 4-layer prune of sgl-project/DeepSeek-V4-Flash-FP8.
    "DeepSeek-V4-Flash-FP8-4layer": "Pinaster",
}

_MEGATRON_MODEL_TYPE = {
    "DeepSeek-V4-Flash-FP8": "deepseek-v4-flash",
    "DeepSeek-V4-Flash-FP8-4layer": "deepseek-v4-flash-4layer",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    # Context parallelism for the training actor. 1 keeps the historical layout untouched.
    # Above 1 it is what makes long sequences fit, because it is the only axis that divides
    # the indexer's [seqlen, seqlen_kv] score matrix -- pipeline parallelism shrinks how many
    # layers a rank owns but every one of them still allocates the matrix in full.
    context_parallel_size: int = 1
    run_id: str = U.create_run_id()
    model_org: str = ""
    model_name: Literal[
        "DeepSeek-V4-Flash-FP8",
        "DeepSeek-V4-Flash-FP8-4layer",
    ] = "DeepSeek-V4-Flash-FP8"

    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    join_ray_workers: bool = True
    ray_hostfile: str = "/root/mpi_rack_hostfile"
    ray_port: int = 6379
    enable_eval: bool = True
    enable_mtp: bool = False
    colocate_memory_profile: Literal["auto", "192gb", "288gb"] = "auto"

    hf_checkpoint: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    # Defaults to model_dir. Set explicitly when shared NFS -> per-node local NVMe copy is needed.
    model_local_dir: str | None = None
    save_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # performance configs
    num_gpus_per_node: int = 8
    # use colocate by default. will switch to disaggregated mode when 0 < rollout_num_nodes < num_nodes
    rollout_num_nodes: int = 0
    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)
    rollout_num_gpus: int = field(init=False)
    optimizer_offload: bool = True
    use_fault_tolerance: bool = True

    # debug configs
    dump_details: bool = False
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    debug_data_root: str = "/root/shared_data"
    skip_saving: bool = False

    # precision configs
    enable_r3: bool = True
    train_deterministic: bool = True
    # TE FP8 GEMMs when True, BF16 only when explicitly disabled. Independent of rollout.
    fp8_training: bool = True
    fp8_recipe: Literal["auto", "blockwise", "tensorwise"] = "auto"
    enable_mis: bool = False

    # pass any extra sglang/miles/megatron args through `--extra-args '--your-arg'`
    extra_args: str = ""

    def __post_init__(self):
        if not self.model_org:
            self.model_org = _DEFAULT_MODEL_ORG[self.model_name]
        if self.model_local_dir is None:
            self.model_local_dir = self.model_dir
        assert self.rollout_num_nodes >= 0
        assert self.rollout_num_nodes < self.num_nodes
        self.colocate = self.rollout_num_nodes == 0
        self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node
        if self.colocate:
            self.rollout_num_gpus = self.num_nodes * self.num_gpus_per_node
        else:
            self.rollout_num_gpus = self.rollout_num_nodes * self.num_gpus_per_node

    @property
    def megatron_model_type(self):
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def torch_dist_name(self):
        return f"{self.model_name}_torch_dist"

    @property
    def bf16_name(self):
        return f"{self.model_name}-bf16"


def _download_dataset(args: ScriptArgs):
    """Download the task-specific dataset(s)."""
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _hf_checkpoint_path(args: ScriptArgs) -> str:
    """Resolve hf_checkpoint path: explicit override wins, else {model_dir}/{model_name}."""
    return args.hf_checkpoint or f"{args.model_dir}/{args.model_name}"


def _ensure_4layer_model_type(args: ScriptArgs):
    """Undo the old deepseek_ref workaround for local 4-layer prunes."""
    if args.model_name != "DeepSeek-V4-Flash-FP8-4layer":
        return
    cfg = Path(_hf_checkpoint_path(args)) / "config.json"
    if not cfg.exists():
        return
    text = cfg.read_text()
    if '"model_type": "deepseek_ref"' in text:
        cfg.write_text(text.replace('"model_type": "deepseek_ref"', '"model_type": "deepseek_v4"'))
        print(f"[patch] {cfg}: model_type deepseek_ref -> deepseek_v4")


def _prepare_download(args: ScriptArgs):
    """Download HF checkpoint + task dataset. Idempotent: hf skips existing blobs."""
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    # Only download if the user has NOT supplied a pre-existing checkpoint dir.
    # (prepare_single / train with --hf-checkpoint bypass this.)
    if args.hf_checkpoint is None:
        dest = f"{args.model_dir}/{args.model_name}"
        U.exec_command_cpu(f"hf download {args.model_org}/{args.model_name} " f"--local-dir {dest}")
    _ensure_4layer_model_type(args)
    _download_dataset(args)


@app.command()
@U.dataclass_cli
def prepare_download(args: ScriptArgs):
    """Download HF checkpoint + dataset from HuggingFace. Run on one node (shared NFS)."""
    _prepare_download(args)


def _prepare_single(args: ScriptArgs):
    _download_dataset(args)

    src = _hf_checkpoint_path(args)
    U.fp8_cast_bf16(
        path_src=src,
        path_dst=f"{args.model_dir}/{args.bf16_name}/",
    )


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    """FP8 -> BF16 cast for Megatron. Needs --hf-checkpoint (or pre-downloaded). One node."""
    _prepare_single(args)


def _prepare_spmd(args: ScriptArgs):
    is_4layer = args.model_name == "DeepSeek-V4-Flash-FP8-4layer"
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    extra_args = "--dsv4-impl miles --expert-tensor-parallel-size 1 --context-parallel-size 1 "
    if actor_num_nodes == 1 and is_4layer:
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        )
    elif actor_num_nodes == 1 and args.model_name == "DeepSeek-V4-Flash-FP8":
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 8 "
        )
    else:
        raise NotImplementedError(
            f"No verified SPMD conversion config for {args.model_name} "
            f"({actor_num_nodes} actor nodes x {actor_num_gpus_per_node} GPUs/node). "
            f"Please specify your conversion parallel config in `run_deepseek_v4.py`."
        )

    num_gpus_for_convert = actor_num_gpus_per_node
    if is_4layer:
        num_gpus_for_convert = min(num_gpus_for_convert, 4)

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.bf16_name}",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_for_convert,
        multinode=True if actor_num_nodes > 1 else False,
        num_nodes=actor_num_nodes,
        extra_args=extra_args,
        dir_dst=f"{args.model_dir}",
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _prepare_spmd(args)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    _prepare_cp(args)


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.torch_dist_name}",
        path_dst=f"{args.model_local_dir}/{args.torch_dist_name}",
        num_nodes=args.num_nodes,
    )
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_local_dir}/{args.model_name}",
        num_nodes=args.num_nodes,
    )


def _resolve_colocate_memory_profile(args: ScriptArgs) -> str:
    """Pick the colocate memory split from the card, not from the node count."""
    if args.colocate_memory_profile != "auto":
        return args.colocate_memory_profile
    try:
        import torch

        if not torch.cuda.is_available():
            return "288gb"
        gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return "288gb"
    return "288gb" if gib >= 256 else "192gb"


def _is_gfx942() -> bool:
    # Hardware probe is delayed until launch, keeping imports GPU-free.
    import torch

    return bool(
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] == "gfx942"
    )


def _cp_derived_layout(
    total_gpus: int, *, pipeline_size: int, first_layers: int, last_layers: int, cp_size: int
) -> str:
    """Spend the tensor-parallel budget on context parallelism, keeping the pipeline split fixed.

    TP = total_gpus / (PP * CP), so DP collapses to 1. EP stays at 8 either way: Megatron builds
    the expert group from world_size % (etp * ep * pp), a quantity CP does not enter.
    """
    if total_gpus % (pipeline_size * cp_size):
        raise NotImplementedError(
            f"--context-parallel-size {cp_size} does not divide {total_gpus} GPUs "
            f"over {pipeline_size} pipeline stages."
        )
    tensor_size = total_gpus // (pipeline_size * cp_size)
    config = f"--tensor-model-parallel-size {tensor_size} "
    if tensor_size > 1:
        # Megatron rejects sequence parallelism at TP=1 rather than ignoring it.
        config += "--sequence-parallel "
    config += (
        f"--pipeline-model-parallel-size {pipeline_size} "
        f"--decoder-first-pipeline-num-layers {first_layers} "
        f"--decoder-last-pipeline-num-layers {last_layers} "
        f"--context-parallel-size {cp_size} "
    )
    if cp_size > 1:
        # Mandatory above CP=1: DeepSeek V4 has no zigzag CP path, and arguments.py asserts on
        # the flag rather than setting it for you.
        config += "--allgather-cp "
    return config + "--expert-model-parallel-size 8 " "--expert-tensor-parallel-size 1 "


def _get_parallel_config(args: ScriptArgs) -> str:
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    total_gpus = actor_num_nodes * actor_num_gpus_per_node

    # Single-node smoke-test configs
    if actor_num_nodes == 1:
        return (
            f"--tensor-model-parallel-size {actor_num_gpus_per_node} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {actor_num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )

    if actor_num_gpus_per_node == 8:
        cp_size = args.context_parallel_size
        if total_gpus == 32:  # 4 nodes x 8 GPUs: PP4/EP8, 43 layers = 11+11+11+10
            if cp_size == 1:
                # The original layout, kept verbatim: TP4/PP4/CP1/EP8, DP2.
                return (
                    "--tensor-model-parallel-size 4 "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 4 "
                    "--decoder-first-pipeline-num-layers 11 "
                    "--decoder-last-pipeline-num-layers 10 "
                    "--context-parallel-size 1 "
                    "--expert-model-parallel-size 8 "
                    "--expert-tensor-parallel-size 1 "
                )
            return _cp_derived_layout(total_gpus, pipeline_size=4, first_layers=11, last_layers=10, cp_size=cp_size)

        if total_gpus == 16:  # 2 nodes x 8 GPUs: PP2/EP8, 43 layers = 22+21
            return _cp_derived_layout(total_gpus, pipeline_size=2, first_layers=22, last_layers=21, cp_size=cp_size)

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} GPUs. "
        f"Please specify your parallel config in `run_deepseek_v4._get_parallel_config`."
    )


@dataclass(frozen=True)
class _FP8Capabilities:
    device: str
    te_version: str
    blockwise: tuple[bool, str]
    tensorwise: tuple[bool, str]


def _probe_fp8_capabilities() -> _FP8Capabilities:
    # TE is optional for download/conversion and importing launchers on CPU CI.
    # Probe only when training is requested, before any Ray jobs are submitted.
    try:
        import torch
        import transformer_engine
        from transformer_engine.common import recipe
        from transformer_engine.pytorch import fp8
    except ImportError as exc:
        raise RuntimeError("FP8 training requires Transformer Engine with FP8 recipe support.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 recipe preflight requires a visible training GPU; cannot verify TE capabilities.")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    device = f"{props.name} ({getattr(props, 'gcnArchName', 'CUDA')})"

    def support(class_name, check_name):
        check = getattr(fp8, check_name, None)
        if not hasattr(recipe, class_name) or check is None:
            return False, f"TE lacks {class_name} / {check_name}"
        return check()

    return _FP8Capabilities(
        device=device,
        te_version=transformer_engine.__version__,
        blockwise=support("Float8BlockScaling", "check_fp8_block_scaling_support"),
        tensorwise=support("Float8CurrentScaling", "check_fp8_support"),
    )


def _resolve_fp8_recipe(requested: str) -> str:
    if requested not in ("auto", "blockwise", "tensorwise"):
        raise ValueError(f"Unknown FP8 recipe: {requested}")
    caps = _probe_fp8_capabilities()
    candidates = ("blockwise", "tensorwise") if requested == "auto" else (requested,)
    for candidate in candidates:
        if getattr(caps, candidate)[0]:
            print(
                f"[precision] FP8 E4M3 requested={requested} selected={candidate}; "
                f"TE={caps.te_version}; device={caps.device}; blockwise_support={caps.blockwise}"
            )
            return candidate
    raise RuntimeError(
        f"FP8 recipe '{requested}' is unsupported on {caps.device} (TE {caps.te_version}). "
        f"blockwise: {caps.blockwise[1]}; tensorwise: {caps.tensorwise[1]}. "
        "Use --fp8-recipe auto or --fp8-recipe tensorwise on a supported TE build. "
        "No BF16 fallback was applied."
    )


def _get_fp8_training_recipe(args: ScriptArgs) -> str | None:
    if not args.fp8_training:
        return None
    if any(token.split("=", 1)[0] == "--fp8-recipe" for token in shlex.split(args.extra_args)):
        raise ValueError("Set the launcher's --fp8-recipe, not --extra-args, so TE support is checked.")
    return _resolve_fp8_recipe(args.fp8_recipe)


def _train(args: ScriptArgs, *, fp8_recipe: str | None = None):
    print(f"[precision] fp8_training={args.fp8_training}")
    if fp8_recipe is None:
        fp8_recipe = _get_fp8_training_recipe(args)
    print(
        f"running on {args.num_nodes} nodes "
        f"({args.actor_num_nodes} actor nodes x {args.actor_num_gpus_per_node} GPUs/node, "
        f"{args.rollout_num_gpus} rollout GPUs, colocate={args.colocate})"
    )
    _ensure_4layer_model_type(args)

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    ckpt_args = f"--hf-checkpoint {_hf_checkpoint_path(args)} " f"--ref-load {args.model_local_dir}/{args.torch_dist_name} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} " f"--save {load_save_path} " "--save-interval 20 " "--save-retain-interval 20 "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len 8192 "
                """--apply-chat-template-kwargs '{"thinking_mode":"thinking"}' """
            )
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 4096 "
            )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    perf_args = _get_parallel_config(args)

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
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
    if args.optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--use-precision-aware-optimizer " "--overlap-cpu-optimizer-d2h-h2d "
        )
        if args.actor_num_nodes == 4:
            if _resolve_colocate_memory_profile(args) == "288gb":
                # 288 GB card: partial optimizer offload (keep ~25% on GPU) + keep train
                # weights on GPU; pair with --sglang-mem-fraction-static 0.5.
                optimizer_args += "--optimizer-offload-fraction 0.75 " "--no-offload-train "
            else:
                optimizer_args += "--optimizer-offload-fraction 1.0 " "--offload-train "

    sglang_world_size = 4
    sglang_tp_size = 4
    sglang_dp_size = 1
    sglang_ep_size = 4
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        f"--sglang-tp-size {sglang_tp_size} "
        f"--sglang-dp-size {sglang_dp_size} "
        f"--sglang-ep-size {sglang_ep_size} "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "  # TODO improve
        # Small GRPO batches share prefixes; cache-aware routing's default
        # imbalance threshold (64) can queue the whole batch on one engine.
        "--sglang-router-policy round_robin"
    )
    if _is_gfx942():
        # gfx942: AITER's BF16 atomic reductions perturb router/expert results
        # between identical forwards. Triton still executes FP8 expert GEMMs.
        sglang_args += " --sglang-moe-runner-backend triton --sglang-disable-custom-all-reduce "
    extra_env_vars = {
        "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
        "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "0",
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
        # unified_kv lives in compressor_v2 only; on HIP the v1 path leaves
        # compress_kv_pool unset and the memory pool asserts on it.
        "SGLANG_OPT_USE_COMPRESSOR_V2": "true",
        "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
        "SGLANG_OPT_USE_JIT_NORM": "true",
        "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "AITER_BF16_FP8_MOE_BOUND": "0",
    }
    if _is_gfx942():
        # Bound per-process hardware queues in colocated training/serving.
        # Keep all overrides in the Ray runtime env, never host configuration.
        extra_env_vars |= {"GPU_MAX_HW_QUEUES": "1", "SGLANG_SET_CPU_AFFINITY": "0"}

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--train-memory-margin-bytes {'3221225472' if _resolve_colocate_memory_profile(args) == '288gb' else '1073741824'} "
        f"--sglang-mem-fraction-static {'0.5' if _resolve_colocate_memory_profile(args) == '288gb' else '0.85'} "
        "--sglang-watchdog-timeout 1800 "  # ROCm: slow aiter gemm tune under colocate; avoid watchdog SIGQUIT
        "--accumulate-allreduce-grads-in-fp32 "
        "--dsv4-impl miles "  # ROCm has no cudnn/flash_mla path for the megatron impl
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format thd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    if args.colocate:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {args.rollout_num_gpus} "

    if args.dump_details:
        misc_args += f"--dump-details {args.debug_data_root}/{args.run_id}/dump_details "

    if args.enable_mis:
        misc_args += (
            "--use-tis "
            "--custom-config-path examples/infra_features/train_infer_mismatch_helper/mis.yaml "
            "--custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "

    if args.debug_train_run_id is not None:
        if args.debug_train_rollout_id is None:
            args.debug_train_rollout_id = 1
        misc_args += (
            f"--load-debug-rollout-data "
            f"{args.debug_data_root}/{args.debug_train_run_id}/dump_details/rollout_data/{args.debug_train_rollout_id}.pt "
        )
        misc_args += "--debug-train-only "

    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "
        # Skip indexer-replay for now
        # misc_args += "--use-rollout-indexer-replay "

    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    if args.fp8_training:
        misc_args += (
            "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " f"--fp8-recipe {fp8_recipe} "
        )
        if fp8_recipe == "blockwise":
            misc_args += """--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"0"}' """
        # ROCm TE MoE FP8 lacks fused wgrad accumulation; disable the fusion.
        misc_args += "--no-gradient-accumulation-fusion "

    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )
        # gfx950: use RCCL all-gather for speculative decoding; aiter can deadlock.
        extra_env_vars |= {"SGLANG_USE_AITER_AG": "false"}

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**extra_env_vars},
        megatron_path=args.megatron_path,
        before_ray_job_submit=(
            (
                lambda: U.ssh_start_ray_workers(
                    master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
                    num_gpus_per_node=args.num_gpus_per_node,
                    hostfile=args.ray_hostfile,
                    port=args.ray_port,
                )
            )
            if args.join_ray_workers and args.num_nodes > 1
            else None
        ),
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training. Assumes data/model/torch_dist are already prepared on {model_local_dir}."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    fp8_recipe = _get_fp8_training_recipe(args)
    _prepare_download(args)

    bf16_dir = Path(f"{args.model_dir}/{args.bf16_name}")
    bf16_sentinel = bf16_dir / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        _prepare_single(args)
    else:
        print(f"[full_train] Skipping FP8->BF16 cast: {bf16_sentinel} already exists.")

    torch_dist_dir = Path(f"{args.model_dir}/{args.torch_dist_name}")
    torch_dist_sentinel = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"[full_train] Skipping BF16->torch_dist conversion: {torch_dist_sentinel} already exists.")

    if args.model_local_dir != args.model_dir:
        _prepare_cp(args)
    else:
        print(f"[full_train] Skipping rsync: model_local_dir == model_dir ({args.model_dir})")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"

    _train(args, fp8_recipe=fp8_recipe)


if __name__ == "__main__":
    app()
