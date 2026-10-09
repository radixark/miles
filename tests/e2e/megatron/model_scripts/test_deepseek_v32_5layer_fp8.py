"""DeepSeek V3.2 5-layer CI smoke test on 4x H200 (2 training GPUs + 2 rollout GPUs).

FP8 rollout using the raw DeepSeek FP8 checkpoint (no re-quantization).
BF16 training via Megatron. Thin wrapper around scripts/run_deepseek_v32.py.
"""

import os

from scripts.run_deepseek_v32 import (
    ScriptArgs,
    _execute_train,
    _prepare_bf16_ckpt,
    _prepare_download,
    _prepare_megatron_ckpt,
)
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(est_time=1300, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    return ScriptArgs.from_env(
        model_org="Pinaster",
        model_name="DeepSeek-V3.2-5layer",
        megatron_model_type="deepseek-v32-5layer",
        hardware="H200",
        use_single_node=True,
        num_gpus_per_node=4,
        from_bf16_ckpt=False,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        no_save=True,
        extra_args=(
            "--ci-test --check-weight-update-allow-quant-error --bf16 "
            "--use-rollout-routing-replay "
            "--skip-actor-forward-only "
            "--sglang-disable-shared-experts-fusion "
            # TP2 doubles the per-rank activations of the script's TP4 token budget
            "--max-tokens-per-gpu 16384 "
        ),
    )


def prepare(args: ScriptArgs):
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_megatron_ckpt(args)


def execute(args: ScriptArgs):
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
