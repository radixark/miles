import os

from examples.lora.run_qwen3_lora_native import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

# Smoke test for the native (raw-mode) LoRA plugin on the dense-GQA reference
# recipe. Qwen3-0.6B on 2 GPUs with TP2+SP is the cheapest end-to-end check and
# exercises both adapter grad-summation paths (column- and row-parallel). Full
# rollout -> train -> adapter-sync loop with the CI checkers on (cross-engine
# logprob agreement at abs_tol 0.03, step-0 ppo_kl).

register_cuda_ci(est_time=1500, suite="stage-c-2-gpu-h200", labels=["lora-native"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Qwen3-0.6B",
        num_nodes=1,
        num_gpus_per_node=2,
        num_rollout=2,
        enable_wandb=False,
        extra_args="--ci-test ",
    )


def prepare(args: ScriptArgs):
    _prepare_download(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    args = _args()
    prepare(args)
    execute(args)
