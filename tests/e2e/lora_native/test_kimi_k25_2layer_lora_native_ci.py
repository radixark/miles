import os

from examples.lora.run_lora_native import ScriptArgs, _prepare, _train
from tests.ci.ci_register import register_cuda_ci

# Native (raw-mode) LoRA on the 2-layer Kimi-K2.5 prune: MLA adapters on the
# multimodal-shelled checkpoint. Prepare covers the whole K2.5 pipeline this
# repo owns — INT4 -> BF16 dequant (which must strip quantization_config, or
# SGLang serves the BF16 weights through its CompressedTensors path with a
# context-free forward), the kimi_k25 mbridge conversion, and the
# language_model-prefixed weight-sync naming. The logprobs checker is disabled
# like the other pruned-toy CIs: the toy's rollout entropy legitimately exceeds
# the checker's real-model bound.

register_cuda_ci(est_time=3600, suite="stage-c-8-gpu-h200", labels=["lora-native"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Kimi-K2.5-2layer",
        model_dir="/root/models",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        enable_wandb=False,
        extra_args=(
            "--ci-test --ci-disable-logprobs-checker " "--check-weight-update-skip-list vision_tower. mm_projector. "
        ),
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
