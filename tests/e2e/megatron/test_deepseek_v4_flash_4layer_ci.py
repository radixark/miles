import os

from scripts.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", labels=["model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="DeepSeek-V4-Flash-FP8-4layer",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=8,
        skip_saving=True,
        use_fault_tolerance=False,
        extra_args=(
            "--ci-test "
            "--check-weight-update-allow-quant-error "
            "--ci-disable-logprobs-checker "
            "--disable-weights-backuper "
            "--num-rollout 2 "
        ),
    )


def prepare(args: ScriptArgs):
    _prepare_download(args)
    _prepare_single(args)
    _prepare_spmd(args)
    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
