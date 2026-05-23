import os

from scripts.run_qwen3_4b import ScriptArgs, execute, prepare
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=1200, suite="stage-c-8-gpu-h100", labels=["megatron", "precision"])


def _clear_proxy_env():
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)


def _build_args() -> ScriptArgs:
    run_id = os.environ.get("GITHUB_COMMIT_NAME") or f"qwen3-dense-top-tp4-cp2-ci-{U.create_run_id()}"
    args = ScriptArgs(
        mode="debug_one_sample",
        run_id=run_id,
        model_name="Qwen3-4B",
        train_backend="megatron",
        true_on_policy=True,
        enable_eval=False,
        # Covers the complementary 8-GPU dense layout: TP=4, CP=2.
        extra_args="--num-rollout 1 --rollout-max-response-len 1024 ",
    )
    args.tensor_model_parallel_size = 4
    args.context_parallel_size = 2
    args.cp_comm_type = "a2a"
    return args


if __name__ == "__main__":
    args = _build_args()
    prepare(args)
    _clear_proxy_env()
    execute(args)
