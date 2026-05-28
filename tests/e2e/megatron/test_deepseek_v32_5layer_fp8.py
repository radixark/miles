"""DeepSeek V3.2 5-layer CI smoke test on H100.

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

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", labels=["model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_org="Pinaster",
        model_name="DeepSeek-V3.2-5layer",
        megatron_model_type="deepseek-v32-5layer",
        hardware="H100",
        use_single_node=True,
        from_bf16_ckpt=False,
        num_rollout=2,
        extra_args=(
            "--ci-test --bf16 --freeze-indexer "
            "--use-rollout-routing-replay --use-miles-router "
            "--sglang-disable-shared-experts-fusion "
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
