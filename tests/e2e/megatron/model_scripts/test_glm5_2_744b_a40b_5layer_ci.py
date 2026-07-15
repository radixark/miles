import os

from scripts.run_glm5_2_744b_a40b import (
    ScriptArgs,
    _convert_to_fp8,
    _execute_train,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for the GLM-5.2 (glm_moe_dsa) training script. Exercises the DSA
# cross-layer index-sharing path (5 layers = 3 dense + 2 MoE, computing layers
# 0,1,2 + skip layers 3,4). Verifies the script is functional, not model accuracy.


register_cuda_ci(est_time=900, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=2,
        enable_optimizer_offload=True,
        extra_args=("--ci-test " "--ci-disable-logprobs-checker " "--disable-weights-backuper "),
    )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.output_dir}")
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)


def execute(args: ScriptArgs):
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
