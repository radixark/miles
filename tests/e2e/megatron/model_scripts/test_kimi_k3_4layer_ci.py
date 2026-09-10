import os

from scripts.run_kimi_k3 import ScriptArgs, _prepare_bf16, _prepare_download, _prepare_torch_dist, _train
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(est_time=3000, suite="stage-c-8-gpu-h200", labels=["megatron", "model-scripts"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Kimi-K3-4layer",
        train_mode="full",
        mode="normal",
        task="gsm8k",
        # the pruned model scores 0 on gsm8k, which zeroes every advantage; a fixed pseudo-random
        # reward keeps the weights moving so the MXFP4 re-quantization carries real deltas
        reward_model="deterministic_random",
        hardware="H200",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=64,
        rollout_max_response_len=256,
        rollout_max_concurrency=16,
        check_weight_update_equal=True,
        skip_saving=True,
        extra_args="--ci-test --check-weight-update-allow-quant-error --ci-disable-logprobs-checker ",
    )


def prepare(args: ScriptArgs):
    _prepare_download(args)
    _prepare_bf16(args)
    _prepare_torch_dist(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
