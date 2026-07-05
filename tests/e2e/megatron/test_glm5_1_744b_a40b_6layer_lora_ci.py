import os

from scripts.run_glm5_1_744b_a40b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for scripts/run_glm5_1_744b_a40b_lora.py (LoRA-via-Megatron-Bridge) on the
# 6-layer toy: full rollout -> train -> save loop with a tiny gsm8k rollout, exercising the
# DSA bridge-LoRA path (indexer excluded from LoRA targets). Verifies the script is
# functional, not model accuracy. Uses 4 of the suite's GPUs (TP=EP=4).


register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", labels=["model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.1-6layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=1,
        enable_wandb=False,
        # The weight-update checker (auto-on under --ci-test) must skip the engine-side stacked
        # params that a frozen-base LoRA run cannot re-ship: sglang stacks q_a+kv_a into
        # fused_qkv_a_proj_with_mqa at load, and the DSA indexer weights likewise have no 1:1
        # trainer export. They keep their (correct) checkpoint values; everything else is verified.
        extra_args=(
            "--ci-test "
            "--ci-disable-logprobs-checker "
            "--disable-weights-backuper "
            "--check-weight-update-skip-list fused_qkv_a_proj_with_mqa indexer. "
        ),
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
