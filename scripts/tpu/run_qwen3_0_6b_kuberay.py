"""Qwen3-0.6B GRPO on gsm8k on a KubeRay TPU v7x cluster: FSDP trainer + vllm-torchtpu rollout, weights over tpu-sync.

The trainer (4 devices) and one vLLM engine (TP 4) run in separate Ray placement groups, so they land on two 2x2x1
slices when two are alive and share one slice as 4+4 sub-host sessions otherwise. Weights move from the trainer's
FSDP shards to the engine's TP shards host to host over TCP (`--update-weight-transfer-mode tpu-sync`), so neither
layout needs the job-local-world torch_tpu build that `--update-weight-transfer-mode ici` does.

Run it from the KubeRay head pod against the cluster's own Ray, with this checkout staged on storage every pod sees
(e.g. GCS FUSE under `/data`) and `MILES_SCRIPT_EXTERNAL_RAY=1` exported: the launcher submits `train.py` by its path
in this checkout. docs/hardware-platforms/google-tpus.md#kuberay-on-gke-tpu7x describes the cluster it expects and how
to get the checkout there.

Args:
  --recipe: smoke (3 small rollouts, no eval) or full (100 rollouts, gsm8k test eval every 10).
  --train-backend: fsdp (default; runs on torch 2.13 images) or torchtitan (needs torchtitan 0.4 and torch >= 2.14).
  --model-dir / --data-dir: directory holding Qwen3-0.6B / directory gsm8k is downloaded into.
  --compile-cache-dir: root for the trainer's and the engine's compile caches, kept across runs and preemptions
    when it is on the shared volume (e.g. /data/tmp/miles-cache); empty keeps them in each node's /tmp.
  --extra-args: more train.py flags, appended last.

Examples:
  MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/tpu/run_qwen3_0_6b_kuberay.py prepare
  MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/tpu/run_qwen3_0_6b_kuberay.py train --recipe smoke \
    --compile-cache-dir /data/tmp/miles-cache
  MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/tpu/run_qwen3_0_6b_kuberay.py train --train-backend torchtitan
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

HF_REPO = "Qwen/Qwen3-0.6B"
MODEL_NAME = "Qwen3-0.6B"
NUM_TRAINER_DEVICES = 4
ENGINE_TP_SIZE = 4
# a v7x 2x2x1 host: 4 chips x 2 TensorCores, each a Ray "TPU" resource
DEVICES_PER_HOST = 8
# prompt plus response; also the static length every packed trainer microbatch is padded to
SEQ_LEN = 2048

# PROMOTE_F32: v7x evaluates bf16 transcendentals (SiLU, sigmoid) in bf16 unless told otherwise, which skews the
# engine's logprobs against the trainer's; the scan emitter miscompiles cumulative ops on libtpu >= 0.0.44
_LIBTPU_ARGS = (
    "--xla_tpu_bf16_emission_mode=PROMOTE_F32 "
    "--xla_tpu_use_enhanced_launch_barrier=false "
    "--xla_tpu_use_associative_scan_emitter=false"
)


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
    "full": _Recipe(100, 32, 8, 1024, 256, enable_eval=True),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    recipe: Literal["smoke", "full"] = "smoke"
    train_backend: Literal["fsdp", "torchtitan"] = "fsdp"
    model_dir: str = "/data/assets/hf"
    data_dir: str = "/data/datasets"
    compile_cache_dir: str = ""
    extra_args: str = ""
    # a SIGKILL of a process holding a TPU can wedge the host's VFIO devices for the next run
    graceful_ray_stop: bool = True


def _assert_external_ray_on_kuberay() -> None:
    # KubeRay's own pods carry its generated `ray start`; restarting Ray there would take the whole cluster down
    if os.environ.get("KUBERAY_GEN_RAY_START_CMD") and not os.environ.get("MILES_SCRIPT_EXTERNAL_RAY"):
        raise RuntimeError(
            "This pod belongs to a KubeRay cluster: export MILES_SCRIPT_EXTERNAL_RAY=1 so the launcher submits to "
            "the running Ray instead of stopping it and starting its own"
        )


def _prepare(args: ScriptArgs) -> None:
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _train_env(args: ScriptArgs) -> dict[str, str]:
    compile_cache = (
        {"TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": args.compile_cache_dir} if args.compile_cache_dir else {}
    )
    return {
        "MILES_ACCELERATOR": "tpu",
        "LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", _LIBTPU_ARGS),  # a caller's flag set wins
        **compile_cache,
        # these store torch_tpu programs without their devices, so a trainer could load the engine's
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        "TPU_STDERR_LOG_LEVEL": os.environ.get("TPU_STDERR_LOG_LEVEL", "1"),
        "TPU_MIN_LOG_LEVEL": os.environ.get("TPU_MIN_LOG_LEVEL", "1"),
        # wandb's system-stats sampler polls the node's TPU device plugin, whose restarts kill TPU sessions
        "WANDB_X_DISABLE_STATS": "true",
    }


def _train_backend_args(args: ScriptArgs) -> str:
    shared = (
        "--gradient-checkpointing "
        "--distributed-backend tpu_dist "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {SEQ_LEN} "
    )
    if args.train_backend == "torchtitan":
        return (
            "--train-backend torchtitan "
            "--titan-model-name qwen3 "
            "--titan-model-flavor 0.6B "
            f"--seq-length {SEQ_LEN} "
            "--micro-batch-size 1 "
            f"{shared}"
        )
    return f"--train-backend fsdp {shared}"


def _execute(args: ScriptArgs) -> None:
    _assert_external_ray_on_kuberay()
    U = args.create_backend()
    recipe = _RECIPES[args.recipe]
    # exported rather than passed as extra_env_vars so execute_train skips its nvidia-smi probe
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    ckpt_args = f"--hf-checkpoint {args.model_dir}/{MODEL_NAME} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        # thinking off: a thinking response often runs out of tokens before its answer, so every reward is 0
        """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
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
            f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            f"--eval-max-response-len {recipe.max_response_len} "
            "--eval-top-p 1 "
        )

    # no KL term: its reference model would cost another compiled pass for a zero coefficient
    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    rollout_engine_args = (
        "--rollout-engine vllm "
        f"--rollout-num-gpus {ENGINE_TP_SIZE} "
        f"--rollout-num-gpus-per-engine {ENGINE_TP_SIZE} "
        f"--vllm-max-model-len {SEQ_LEN} "
        "--vllm-gpu-memory-utilization 0.7 "
        "--vllm-max-num-seqs 64 "
        "--update-weight-transfer-mode tpu-sync "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_TRAINER_DEVICES} "
        f"--num-gpus-per-node {DEVICES_PER_HOST} "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{eval_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
            f"{rollout_engine_args} "
            f"{_train_backend_args(args)} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        num_gpus_per_node=DEVICES_PER_HOST,
        megatron_model_type=None,
        extra_env_vars=_train_env(args),
    )


app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Download Qwen3-0.6B and gsm8k into --model-dir and --data-dir."""
    _prepare(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Prepare as `prepare` does, then submit the run to the cluster's Ray."""
    _prepare(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
