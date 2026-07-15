import os

from scripts.run_glm5_744b_a40b import (
    ScriptArgs,
    _execute_train,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Basic smoke test that exercises the rollout indexer-topk replay path on
# GLM-5 (every layer has an indexer). Enabling --use-rollout-indexer-replay
# also forces --use-indexer-replay via miles_validate_args, so the training
# side consumes the per-layer topk emitted by SGLang.

register_cuda_ci(
    est_time=1400,
    suite="stage-c-2-gpu-h200",
    labels=["megatron", "model-scripts", "replay"],
)


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5_4layer",
        num_nodes=1,
        num_gpus_per_node=2,
        num_rollout=2,
        enable_optimizer_offload=True,
        extra_env_vars="MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1",
        extra_args=(
            "--ci-test "
            "--ci-disable-logprobs-checker "
            "--disable-weights-backuper "
            "--use-rollout-indexer-replay "
            "--rollout-max-response-len 4096 "
            # preserve to avoid CPU OOM
            "--sglang-max-total-tokens 1900000 "
            # exercise indexer replay with two-way tensor and expert parallelism
            "--tensor-model-parallel-size 2 "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 2 "
        ),
    )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.output_dir}")
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
