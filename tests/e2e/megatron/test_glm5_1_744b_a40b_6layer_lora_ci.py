import os

from scripts.run_glm5_1_744b_a40b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for the GLM-5.1 LoRA-via-Megatron-Bridge training script
# (scripts/run_glm5_1_744b_a40b_lora.py, --megatron-to-hf-mode bridge). Runs the full
# rollout -> train -> save loop on the 6-layer toy (jybsuper/GLM-5.1-6layer,
# 3 dense + 3 MoE) with a tiny gsm8k rollout. Exercises the DSA bridge-LoRA path
# (MLA + lightning indexer, indexer excluded from LoRA targets). Verifies the
# script is functional, not model accuracy. Uses 4 of the suite's GPUs (TP=EP=4,
# the validated single-node layout; bshd + micro-batch-size 1 for DSA).


register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", labels=["model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.1-6layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=1,
        enable_wandb=False,
        extra_args=("--ci-test --ci-disable-logprobs-checker --disable-weights-backuper "),
    )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.output_dir}")
    _prepare_download(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
