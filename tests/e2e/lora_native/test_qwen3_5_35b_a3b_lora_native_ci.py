import os

from examples.lora.run_lora_native import ScriptArgs, _prepare, _train
from tests.ci.ci_register import register_cuda_ci

# Native (raw-mode) LoRA on Qwen3.5-35B-A3B: the hybrid GQA+GDN registry —
# gated fused-QKV adapters on the full-attention layers, mixer-only layers
# legitimately carrying none. Covers prepare (download + torch_dist
# conversion) and the full rollout -> train -> adapter-sync loop with the CI
# checkers on. Also guards the raw-mode GDN backward: the registry's historical
# instability note surfaces as grad_norm explosions from step 1 if it regresses.

register_cuda_ci(est_time=3600, suite="stage-c-8-gpu-h200", labels=["lora-native"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Qwen3.5-35B-A3B",
        model_dir="/root/models",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        enable_wandb=False,
        extra_args="--ci-test --check-weight-update-skip-list visual. ",
    )


def prepare(args: ScriptArgs):
    _prepare(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    args = _args()
    prepare(args)
    execute(args)
