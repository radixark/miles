"""What the TPU launchers share: GRPO on one TPU v7x 2x2x1 host, torchtitan trainer + sglang-torchtpu rollout.

The host's 8 TensorCores form one slice that the two jobs share. sglang serves the policy on devices 4-7 (chips 2-3);
the torchtitan trainer runs as 4 ranks on devices 0-3 (chips 0-1) and drives that server as an external rollout
engine. Weights reach the server device to device over ICI (`--update-weight-transfer-mode ici`).
scripts/tpu/README.md describes the environment both expect.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from miles.utils.external_utils import command_utils

SEQ_LEN = 8192
NUM_TRAINER_DEVICES = 4
NUM_ENGINE_DEVICES = 4
SGLANG_BASE_DEVICE = 4
SLICE_SIZE = NUM_TRAINER_DEVICES + NUM_ENGINE_DEVICES
WHOLE_HOST_TOPOLOGY = "2,2,1,2"
# TpuPlatform.weight_transfer_modes; the launcher does not import torch_tpu to read it
WEIGHT_TRANSFER_MODE = "ici"


@dataclass(frozen=True)
class TitanTpuModel:
    """What a launcher fixes about its model; the rest is the shared recipe."""

    # the checkpoint's directory under --model-dir
    name: str
    titan_model_name: str
    titan_model_flavor: str
    sglang_args: str = ""
    train_args: str = ""


@dataclass(frozen=True)
class _Recipe:
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    max_response_len: int
    global_batch_size: int
    enable_eval: bool


_RECIPES: dict[str, _Recipe] = {
    "smoke": _Recipe(3, 8, 4, 7168, 32, enable_eval=False),
    "bench": _Recipe(4, 16, 8, 7168, 128, enable_eval=False),
    "full": _Recipe(100, 16, 8, 7168, 128, enable_eval=True),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    recipe: Literal["smoke", "bench", "full"] = "full"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    sglang_port: int = 30000
    # sglang-torchtpu defaults to fp8 without scales, whose logprobs drift from the bf16 trainer's
    kv_cache_dtype: str = "bfloat16"
    sglang_env_script: str = "/sgl-workspace/sglang/benchmark/tpu/common_env.sh"
    slice_port_base: int = 30100
    extra_args: str = ""
    # the engine shares this host's TPU session, so a run must not SIGKILL anything holding a TPU
    graceful_ray_stop: bool = True
    # keeps the trainer's compiled programs across runs (torch_tpu's tier-3 cache); a rerun's step 0 then loads them
    compile_cache_dir: str = ""


# the associative-scan emitter off: libtpu >= 0.0.44 miscompiles cumulative ops for lengths not a multiple of 1024
_SCAN_EMITTER_OFF = "--xla_tpu_use_associative_scan_emitter=false"
# sglang-torchtpu's benchmark/tpu/common_env.sh flags (exporting LIBTPU_INIT_ARGS replaces them there), with scoped
# VMEM at the v7x maximum
_ENGINE_LIBTPU_ARGS = (
    "--xla_tpu_scoped_vmem_limit_kib=65472 "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_enable_latency_hiding_scheduler=true "
    "--xla_latency_hiding_scheduler_rerun=10 "
    "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative "
    "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative "
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false "
    "--xla_tpu_offload_gather_to_sparsecore=false "
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false "
    f"{_SCAN_EMITTER_OFF}"
)
# the engine's other flags change neither the trainer's compile time nor its step time
_TRAINER_LIBTPU_ARGS = _SCAN_EMITTER_OFF


def _shared_slice(args: ScriptArgs, rank_offset: int) -> str:
    """The shared-slice spec of a job whose ranks start at `rank_offset`; the store sits after the builders."""
    return f"{rank_offset},{SLICE_SIZE},localhost:{args.slice_port_base + SLICE_SIZE}"


def _clear_chip_masks() -> None:
    # both jobs address the whole host's slice; a chip mask hangs libtpu's session setup
    for key in ("TPU_VISIBLE_CHIPS", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"):
        os.environ.pop(key, None)


def _slice_env(args: ScriptArgs, libtpu_args: str) -> dict[str, str]:
    """What both jobs share: one topology and the slice builders; each job brings its own libtpu flags."""
    return {
        "TORCH_TPU_TOPOLOGY": WHOLE_HOST_TOPOLOGY,
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
            f"localhost:{args.slice_port_base + index}" for index in range(SLICE_SIZE)
        ),
        "LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", libtpu_args),  # a caller's flag set wins
        # the engine compiles each program itself; the trainer shares its programs across ranks (_trainer_env)
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled",
        # these store torch_tpu programs without their devices: a 4-rank job would load the 8-device engine's
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        "TPU_STDERR_LOG_LEVEL": os.environ.get("TPU_STDERR_LOG_LEVEL", "1"),
        "TPU_MIN_LOG_LEVEL": os.environ.get("TPU_MIN_LOG_LEVEL", "1"),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1"),
    }


def serve_sglang(args: ScriptArgs, model: TitanTpuModel) -> None:
    """Serve the policy on devices 4-7 in the foreground."""
    U = args.create_backend()
    _clear_chip_masks()
    os.environ.update(_slice_env(args, _ENGINE_LIBTPU_ARGS))
    U.exec_command_cpu(
        f"source {args.sglang_env_script} && "
        f"SGLANG_TPU_SHARED_SLICE={_shared_slice(args, rank_offset=NUM_TRAINER_DEVICES)} "
        "python3 -m sglang.launch_server "
        f"--model-path {args.model_dir}/{model.name} "
        "--device tpu "
        "--dtype bfloat16 "
        "--grammar-backend none "
        # disables the piecewise prefill graph, which TPU does not implement
        "--enable-torch-compile "
        "--page-size 64 "
        f"--kv-cache-dtype {args.kv_cache_dtype} "
        f"--context-length {SEQ_LEN} "
        "--chunked-prefill-size 2048 "
        f"--tp-size {NUM_ENGINE_DEVICES} "
        f"--base-gpu-id {SGLANG_BASE_DEVICE} "
        "--gpu-id-step 1 "
        # the engine has devices 4-7 to itself; the rest of HBM holds XLA's program buffers
        "--mem-fraction-static 0.7 "
        f"{model.sglang_args}"
        f"--host 0.0.0.0 --port {args.sglang_port}"
    )


def prepare_model_and_data(args: ScriptArgs, model_commands: Sequence[str]) -> None:
    """Make the checkpoint with `model_commands` (each skips what is already there), then fetch the datasets."""
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    for command in model_commands:
        U.exec_command_cpu(command)
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _trainer_env(args: ScriptArgs) -> dict[str, str]:
    compile_cache = (
        {"TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": args.compile_cache_dir} if args.compile_cache_dir else {}
    )
    return {
        **_slice_env(args, _TRAINER_LIBTPU_ARGS),
        # one rank compiles each program and the others load it; torch_tpu's tier-3 cache needs this tier on
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "miles_trainer",
        **compile_cache,
        "MILES_ACCELERATOR": "tpu",
        "MILES_TPU_SHARED_SLICE": _shared_slice(args, rank_offset=0),
        # wandb's system-stats sampler polls the node's TPU device plugin, whose restarts kill TPU sessions
        "WANDB_X_DISABLE_STATS": "true",
    }


def execute_train(args: ScriptArgs, model: TitanTpuModel, *, launcher_file: str) -> None:
    """Train on devices 0-3 against the sglang server."""
    U = args.create_backend()
    recipe = _RECIPES[args.recipe]
    _clear_chip_masks()
    # exported rather than passed as extra_env_vars so execute_train skips its nvidia-smi probe
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    model_path = f"{args.model_dir}/{model.name}"
    ckpt_args = f"--hf-checkpoint {model_path}"

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        # thinking off: a thinking response often runs out of tokens before its answer, so every reward is 0
        """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        # grades the last \boxed{}; deepscaler grades only after </think>, which a thinking-off response lacks
        "--rm-type math "
        f"--num-rollout {recipe.num_rollout} "
        f"--rollout-batch-size {recipe.rollout_batch_size} "
        f"--n-samples-per-prompt {recipe.n_samples_per_prompt} "
        f"--rollout-max-response-len {recipe.max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {recipe.global_batch_size} "
    )

    eval_args = ""
    if recipe.enable_eval:
        eval_args = (
            "--eval-interval 10 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 4 "
            f"--eval-max-response-len {recipe.max_response_len} "
            "--eval-top-p 1 "
        )

    # no KL term: its reference model would cost a third compiled pass for a zero coefficient
    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-external-engine-addrs http://localhost:{args.sglang_port} "
        f"--rollout-num-gpus {NUM_ENGINE_DEVICES} "
        f"--rollout-num-gpus-per-engine {NUM_ENGINE_DEVICES} "
        f"--update-weight-transfer-mode {WEIGHT_TRANSFER_MODE} "
    )

    train_backend_args = (
        "--train-backend torchtitan "
        f"--titan-model-name {model.titan_model_name} "
        f"--titan-model-flavor {model.titan_model_flavor} "
        f"--seq-length {SEQ_LEN} "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        "--distributed-backend tpu_dist "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {SEQ_LEN} "
        f"{model.train_args}"
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_TRAINER_DEVICES} "
        f"--num-gpus-per-node {NUM_TRAINER_DEVICES} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{command_utils.get_default_wandb_args(launcher_file, run_id=args.run_id)} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_TRAINER_DEVICES,
        megatron_model_type=None,
        extra_env_vars=_trainer_env(args),
    )
