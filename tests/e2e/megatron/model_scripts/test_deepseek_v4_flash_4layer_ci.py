import fcntl
import os
from pathlib import Path

if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm":
    from scripts.amd.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train
else:
    from scripts.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(est_time=1900, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts"])
register_rocm_ci(
    est_time=1900,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "model-scripts", "amd"],
    disabled="FIXME: re-enable once this case passes on the MI350 runners.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="DeepSeek-V4-Flash-FP8-4layer",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=4,
        skip_saving=True,
        use_fault_tolerance=False,
        extra_args=(
            "--ci-test " "--check-weight-update-allow-quant-error " "--ci-disable-logprobs-checker " "--num-rollout 2 "
        ),
    )


def prepare(args: ScriptArgs):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / f".{args.model_name}.ci-prepare.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
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
