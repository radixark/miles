"""E2E: the GLM-5.2 5-layer smoke test of ``test_glm5_2_744b_a40b_5layer_ci.py`` on 2 Blackwell GPUs with the Cake
dense GEMMs of the miles-ext-kernels submodule: ``--batched-projection-gemm-backend cake`` (the MLA q-absorb and value
projections of the DSA layers) and ``--moe-router-gemm-backend cake`` (the FP32 gate GEMM of the MoE routers). Needs
SM100a / SM103a GPUs and the checked-out submodule.

``--dsa-attention-backend cake`` is not on: its ``det`` kernels build through torch's C++ extensions on first use,
which CI does not cache across jobs.
"""

import dataclasses
import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.model_scripts import test_glm5_2_744b_a40b_5layer_ci as base

register_cuda_ci(
    est_time=1500,
    suite="stage-c-4-gpu-b200",
    labels=["megatron", "model-scripts"],
    hardware=["blackwell"],
    num_gpus=2,
)

CAKE_ARGS = "--use-miles-ext-kernels --batched-projection-gemm-backend cake --moe-router-gemm-backend cake "


def _args():
    args = base._args()
    return dataclasses.replace(args, hardware="B200", extra_args=args.extra_args + CAKE_ARGS)


if __name__ == "__main__":
    args = _args()
    base.prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    base.execute(args)
