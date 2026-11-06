"""E2E: the DeepSeek-V4-Flash 4-layer smoke test of ``test_deepseek_v4_flash_4layer_megatron_impl_ci.py`` on 4
Blackwell GPUs with the Cake mHC kernels of the miles-ext-kernels submodule: ``--mhc-backend cake`` (``det``)
replaces Megatron's hyper connections of every layer. Needs SM100a / SM103a GPUs and the checked-out submodule.
"""

import dataclasses
import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.model_scripts import test_deepseek_v4_flash_4layer_megatron_impl_ci as base

register_cuda_ci(
    est_time=1800,
    suite="stage-c-4-gpu-b200",
    labels=["megatron", "model-scripts"],
    hardware=["blackwell"],
    num_gpus=4,
)

CAKE_ARGS = "--use-miles-ext-kernels --mhc-backend cake --cake-mhc-variant det "


def _args():
    args = base._args()
    return dataclasses.replace(args, hardware="B200", extra_args=args.extra_args + CAKE_ARGS)


if __name__ == "__main__":
    args = _args()
    base.prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    base.execute(args)
