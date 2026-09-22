"""E2E: GLM-5.2 (5-layer prune) full-parameter GRPO with the deterministic DSA kernels.

Runs ``--dsa-attention-backend loom`` (the generated deterministic indexer + sparse-attention kernels
in ``miles_plugins/models/dsa_train``) on the GLM-5.2 training script with TP4 (4 GPUs) and the DSA
cross-layer index-sharing path (5 layers = 3 dense + 2 MoE; computing layers 0-2, skip layers 3-4).
``MILES_DSA_BACKEND=tilelang`` runs the same recipe on the fused TileLang kernels for an A/B reference.
Needs Blackwell (SM100a / SM103a) GPUs for the loom backend.

``MILES_E2E_MODE`` selects ``live`` (default: rollouts + training), ``record`` (live, additionally dumps every
rollout batch to ``MILES_E2E_DEBUG_DIR``) or ``replay`` (``--debug-train-only`` on the recorded batches, saving
the per-step grad norm and the per-rank train data, incl. the trainer log-probs, under ``MILES_E2E_RUN_TAG``).
Two ``replay`` runs of the loom backend on the same recorded batches must produce bit-identical grad norms and
log-probs; ``tilelang`` replays give the non-deterministic reference.

Paths default to the CI layout (``/root/models``, ``/root/datasets``, ``/root/Megatron-LM``); set
``MILES_E2E_ROOT`` to relocate models/datasets/outputs under one directory (out-of-CI runs on scratch storage).
"""

import os

from scripts.run_glm5_2_744b_a40b import (
    ScriptArgs,
    _execute_train,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=1800, suite="stage-c-4-gpu-b200", labels=["megatron", "model-scripts"], hardware=["blackwell"])

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
DEBUG_DIR = os.environ.get("MILES_E2E_DEBUG_DIR", f"{ROOT or '/root'}/glm5_dsa_debug_rollouts")


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
        )
    raise ValueError(f"MILES_E2E_MODE must be live, record or replay, got {MODE!r}")


def _args() -> ScriptArgs:
    kwargs = {}
    if ROOT:
        kwargs.update(
            data_dir=f"{ROOT}/datasets",
            model_dir=f"{ROOT}/models",
            model_local_dir=f"{ROOT}/models",
            output_dir=f"{ROOT}/shared_data",
            megatron_path=os.environ.get("MILES_E2E_MEGATRON_PATH", "/root/Megatron-LM"),
        )
    if os.environ.get("MILES_E2E_NO_DEEPEP") == "1":
        # Megatron's DeepEP does not run on GB300 (see scripts/run_glm5_2_744b_a40b.py).
        kwargs.update(use_deepep=False, megatron_use_deepep=False)
    return ScriptArgs(
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=NUM_ROLLOUT,
        enable_optimizer_offload=True,
        extra_args=(
            "--ci-test --ci-disable-logprobs-checker "
            f"--dsa-attention-backend {BACKEND} "
            + _mode_args()
        ),
        **kwargs,
    )


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    _prepare_megatron_ckpt(args)


def execute(args: ScriptArgs):
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
