import os

from scripts.run_kimi_k25 import ScriptArgs, _convert_to_bf16, _execute_train, _prepare_download
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for the Kimi-K2.5 (MoE + MLA, INT4 rollout + BF16 Megatron bridge) training
# script. It runs the 4-layer pruned model on a single 8-GPU node and only verifies that
# the training script is functional, not model accuracy.


register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", labels=["model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Kimi-K2.5-4layer",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        extra_args=("--ci-test " "--ci-disable-logprobs-checker " "--disable-weights-backuper "),
    )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.output_dir}")
    _prepare_download(args)
    _convert_to_bf16(args)


def execute(args: ScriptArgs):
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
