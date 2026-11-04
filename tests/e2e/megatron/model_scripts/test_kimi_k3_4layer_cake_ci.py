"""E2E: the Kimi-K3 4-layer smoke test of ``test_kimi_k3_4layer_ci.py`` on 8 Blackwell GPUs with the Cake KDA
kernels of the miles-ext-kernels submodule: ``--kda-backend cake`` (12 KDA heads per rank at TP 8) and
``--linear-attention-conv-backend cake`` (the KDA short conv, ``det``). Needs SM100a / SM103a GPUs and the
checked-out submodule.
"""

import dataclasses
import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.model_scripts import test_kimi_k3_4layer_ci as base

register_cuda_ci(
    est_time=1800,
    suite="stage-c-8-gpu-b200",
    labels=["megatron", "model-scripts"],
    hardware=["blackwell"],
    num_gpus=8,
)

CAKE_ARGS = (
    "--use-miles-ext-kernels --kda-backend cake --linear-attention-conv-backend cake --cake-conv1d-variant det "
)


def _args():
    args = base._args()
    return dataclasses.replace(args, hardware="B200", extra_args=args.extra_args + CAKE_ARGS)


if __name__ == "__main__":
    args = _args()
    base.prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    base.execute(args)
