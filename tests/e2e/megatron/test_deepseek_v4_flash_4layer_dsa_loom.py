"""E2E: DeepSeek-V4-Flash (4-layer prune) GRPO with the deterministic DSA kernels.

Runs ``--dsv4-impl miles --dsa-attention-backend loom`` (the generated deterministic kernels in
``miles_plugins/models/dsa_train``: batched ``sbhd`` indexer in one launch, ``bshd`` sparse attention with
the FP32 attention sink) on the DeepSeek-V4 training script with 4 GPUs (TP1, EP1).
``MILES_DSA_BACKEND=tilelang`` runs the same recipe on the per-sample TileLang kernels for an A/B reference.
Needs Blackwell (SM100a / SM103a) GPUs for the loom backend.

``MILES_E2E_MODE`` selects ``live`` (default), ``record`` (live + dump every rollout batch to ``MILES_E2E_DEBUG_DIR``)
or ``replay`` (``--debug-train-only`` on the recorded batches, saving per-step grad norms and per-rank train data
under ``MILES_E2E_RUN_TAG``; ``MILES_E2E_REPLAY_LOSS_ARGS`` defaults to ``--entropy-coef 0.001`` so the
backward carries non-zero gradients although the truncated smoke responses give zero advantages).  Two loom replays must be bit-identical.

Paths default to the CI layout (``/root/models``, ``/root/datasets``, ``/root/Megatron-LM``); set
``MILES_E2E_ROOT`` to relocate them (out-of-CI runs on scratch storage).
"""

import fcntl
import os
from pathlib import Path

from scripts.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(est_time=1900, suite="stage-c-4-gpu-b200", labels=["megatron", "model-scripts"], hardware=["blackwell"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

BACKEND = os.environ.get("MILES_DSA_BACKEND", "loom")
ROOT = os.environ.get("MILES_E2E_ROOT")
MODE = os.environ.get("MILES_E2E_MODE", "live")
NUM_ROLLOUT = int(os.environ.get("MILES_E2E_NUM_ROLLOUT", "2"))
RUN_TAG = os.environ.get("MILES_E2E_RUN_TAG", BACKEND)
DEBUG_DIR = os.environ.get("MILES_E2E_DEBUG_DIR", f"{ROOT or '/root'}/dsv4_dsa_debug_rollouts")


def _mode_args() -> str:
    if MODE == "live":
        return ""
    if MODE == "record":
        os.makedirs(DEBUG_DIR, exist_ok=True)
        return f"--save-debug-rollout-data {DEBUG_DIR}/rollout_{{rollout_id}}.pt "
    if MODE == "replay":
        return (
            f"--load-debug-rollout-data {DEBUG_DIR}/rollout_{{rollout_id}}.pt "
            "--debug-train-only "
            f"--ci-save-grad-norm {DEBUG_DIR}/grad_norm_{RUN_TAG}_{{rollout_id}}_{{step_id}}.pt "
            f"--save-debug-train-data {DEBUG_DIR}/train_{RUN_TAG}_{{rollout_id}}_{{rank}}.pt "
            # The smoke recipes truncate every response, so all advantages (and the policy gradient) are zero;
            # a small entropy term keeps the backward non-trivial so the replayed grad norms compare real kernels.
            + os.environ.get("MILES_E2E_REPLAY_LOSS_ARGS", "--entropy-coef 0.001 ")
        )
    raise ValueError(f"MILES_E2E_MODE must be live, record or replay, got {MODE!r}")


def _args() -> ScriptArgs:
    kwargs = {}
    if ROOT:
        kwargs.update(
            data_dir=f"{ROOT}/datasets",
            model_dir=f"{ROOT}/models",
            save_dir=f"{ROOT}/models",
            debug_data_root=f"{ROOT}/shared_data",
            output_dir=f"{ROOT}/shared_data",
            megatron_path=os.environ.get("MILES_E2E_MEGATRON_PATH", "/root/Megatron-LM"),
        )
    return ScriptArgs(
        model_name="DeepSeek-V4-Flash-FP8-4layer",
        dsv4_impl="miles",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=4,
        skip_saving=True,
        use_fault_tolerance=False,
        extra_args=(
            "--ci-test --check-weight-update-allow-quant-error --ci-disable-logprobs-checker "
            f"--num-rollout {NUM_ROLLOUT} "
            f"--dsa-attention-backend {BACKEND} "
            + _mode_args()
        ),
        **kwargs,
    )


def prepare(args: ScriptArgs):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / f".{args.model_name}.ci-prepare.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _prepare_download(args)
        _prepare_single(args)
        _prepare_spmd(args)
    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
