import fcntl
import os
from pathlib import Path

from scripts.run_deepseek_v4_1 import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(
    est_time=1900,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper", "blackwell"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    return ScriptArgs.from_env(
        model_name="DeepSeek-V4.1-4layer",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=8,
        hardware="H200",
        load_from_hf=True,
        skip_saving=True,
        use_fault_tolerance=False,
        optimizer_offload=True,
        recompute="full",
        extra_args=(
            "--ci-test "
            "--ci-disable-kl-checker "
            "--check-weight-update-allow-quant-error "
            "--num-rollout 2 "
            "--no-pin-cpu-grads "
            "--no-pin-cpu-params "
        ),
    )


def prepare(args: ScriptArgs):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / f".{args.model_name}.ci-prepare.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _prepare_download(args)
    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_dir}/{args.model_name}"


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
