"""Qwen3-4B GRPO on one TPU v7x 2x2x1 host: torchtitan trainer + sglang-torchtpu rollout.

The host's 8 TensorCores are split in half. `sglang` serves the policy on devices 4-7 (chips 2-3);
`train` runs the torchtitan trainer as 4 ranks on devices 0-3 (chips 0-1) and drives that server as an
external rollout engine. Weights reach the server through shared memory (`--weight-transfer shm`, the
default) or a shared filesystem (`disk`). Run `sglang` in its own shell (it stays in the foreground), wait
for it to report healthy, then run `train`. scripts/tpu/README.md describes the environment both expect.

Args:
  --recipe: smoke (3 short rollouts), bench (4 full-shape rollouts, no eval) or full.
  --model-dir / --data-dir: directories holding Qwen3-4B / dapo-math-17k and aime-2024.
  --sglang-port: port of the sglang server `train` connects to.
  --sglang-env-script: sglang-torchtpu's environment script, sourced before the server starts.
  --weight-transfer: shm (default) or disk.
  --weight-dir: where `disk` writes the weights.
  --wandb-project: wandb project; empty disables wandb (the key comes from WANDB_API_KEY or ~/.netrc).
  --compile-cache-dir: persistent cache of the trainer's compiled XLA programs, reused by later runs with
    the same graphs and shapes (a shared filesystem shares it across hosts); empty (default) disables it.
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/tpu/run_qwen3_4b_torchtitan.py sglang
  python scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full
"""

import hashlib
import importlib.metadata
import os
import socket
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3-4B"
SEQ_LEN = 8192
NUM_TRAINER_DEVICES = 4
NUM_ENGINE_DEVICES = 4
SGLANG_BASE_DEVICE = 4
HALF_HOST_TOPOLOGY = "1,2,1,2"


@dataclass(frozen=True)
class _Recipe:
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    max_response_len: int
    global_batch_size: int
    enable_eval: bool


_RECIPES: dict[str, _Recipe] = {
    "smoke": _Recipe(3, 8, 4, 1024, 32, enable_eval=False),
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
    sglang_env_script: str = "/sgl-workspace/sglang/benchmark/tpu/common_env.sh"
    weight_dir: str = "/dev/shm/miles_weights"
    weight_transfer: Literal["shm", "disk"] = "shm"
    wandb_project: str = "miles-tpu"
    compile_cache_dir: str = ""
    extra_args: str = ""


def _free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    for sock in sockets:
        sock.bind(("localhost", 0))
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    return ports


def _set_half_host_env() -> None:
    # The trainer and sglang each address a 2-chip sub-slice; a chip mask hangs libtpu's session setup.
    for key in ("TPU_VISIBLE_CHIPS", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"):
        os.environ.pop(key, None)
    os.environ["TORCH_TPU_TOPOLOGY"] = HALF_HOST_TOPOLOGY
    os.environ.setdefault("TPU_STDERR_LOG_LEVEL", "1")
    os.environ.setdefault("TPU_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def _serve_sglang(args: ScriptArgs):
    U = args.create_backend()
    _set_half_host_env()
    U.exec_command_cpu(
        f"source {args.sglang_env_script} && "
        "SGLANG_TPU_PD_COLOCATED=1 python3 -m sglang.launch_server "
        f"--model-path {args.model_dir}/{MODEL_NAME} "
        "--device tpu "
        "--dtype bfloat16 "
        "--grammar-backend none "
        # Required: it disables the piecewise prefill graph, which TPU does not implement.
        "--enable-torch-compile "
        "--page-size 64 "
        f"--context-length {SEQ_LEN} "
        "--chunked-prefill-size 2048 "
        f"--tp-size {NUM_ENGINE_DEVICES} "
        f"--base-gpu-id {SGLANG_BASE_DEVICE} "
        "--gpu-id-step 1 "
        "--mem-fraction-static 0.4 "
        f"--host 0.0.0.0 --port {args.sglang_port}"
    )


def _prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


# sglang's TPU scheduling flags, with scoped VMEM at the v7x maximum and the associative-scan emitter off
# (libtpu >= 0.0.44 miscompiles cumulative ops for lengths not a multiple of 1024)
_TRAINER_LIBTPU_ARGS = (
    "--xla_tpu_scoped_vmem_limit_kib=65472 "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_enable_latency_hiding_scheduler=true "
    "--xla_latency_hiding_scheduler_rerun=10 "
    "--xla_tpu_use_associative_scan_emitter=false"
)


def _compile_cache_root(cache_dir: str) -> str:
    """torch_tpu keys a cached program by graph, shapes, XLA flags and devices, under a directory per libtpu.

    Its own lowering code is not part of that key, so each torch_tpu build (the hash of its wheel's file
    list) gets a directory of its own.
    """
    torch_tpu = importlib.metadata.distribution("torch_tpu")
    build = hashlib.sha1(torch_tpu.read_text("RECORD").encode()).hexdigest()[:10]
    return f"{cache_dir}/torch_tpu-{torch_tpu.version}-{build}"


def _execute(args: ScriptArgs):
    U = args.create_backend()
    recipe = _RECIPES[args.recipe]
    _set_half_host_env()
    os.environ["MILES_ACCELERATOR"] = "tpu"
    os.environ.setdefault("LIBTPU_INIT_ARGS", _TRAINER_LIBTPU_ARGS)  # a caller's flag set wins
    if args.compile_cache_dir:
        # read once per process at its first compile; ray workers inherit it through `ray start`
        os.environ.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", _compile_cache_root(args.compile_cache_dir))
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        f"localhost:{port}" for port in _free_ports(NUM_TRAINER_DEVICES)
    )
    # wandb's system-stats sampler polls the node's TPU device plugin, whose restarts kill TPU sessions.
    os.environ.setdefault("WANDB_X_DISABLE_STATS", "true")
    # exported rather than passed as extra_env_vars so execute_train skips its nvidia-smi probe
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    model_path = f"{args.model_dir}/{MODEL_NAME}"
    ckpt_args = f"--hf-checkpoint {model_path} --ref-load {model_path} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
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

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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

    # No --wandb-key: the key would land in the job's command line.
    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.run_id} " if args.wandb_project else ""
    )

    sglang_args = (
        f"--rollout-external-engine-addrs http://localhost:{args.sglang_port} "
        f"--rollout-num-gpus {NUM_ENGINE_DEVICES} "
        f"--rollout-num-gpus-per-engine {NUM_ENGINE_DEVICES} "
        f"--update-weight-transfer-mode {args.weight_transfer} "
    )
    if args.weight_transfer == "disk":
        sglang_args += f"--update-weight-disk-dir {args.weight_dir} "

    train_backend_args = (
        "--train-backend torchtitan "
        "--titan-model-name qwen3 "
        "--titan-model-flavor 4B "
        f"--seq-length {SEQ_LEN} "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        "--distributed-backend tpu_dist "
        "--titan-fsdp-gather-once "
        "--use-dynamic-batch-size "
        # headroom under the fixed shape: rebalancing the packed microbatches can exceed the packing budget
        f"--max-tokens-per-gpu {SEQ_LEN - 1024} "
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
        f"{wandb_args} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_TRAINER_DEVICES,
        megatron_model_type=None,
    )


app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def sglang(args: ScriptArgs):
    """Serve the policy on devices 4-7 in the foreground."""
    _serve_sglang(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Download the model and datasets, then train on devices 0-3 against the sglang server."""
    _prepare(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
