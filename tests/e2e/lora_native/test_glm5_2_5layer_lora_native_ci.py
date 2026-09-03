import os

from examples.lora.run_glm5_2_744b_a40b_lora_native import ScriptArgs, _prepare, _train
from tests.ci.ci_register import register_cuda_ci

# Native (raw-mode) LoRA on the 5-layer GLM-5.2 prune (3 dense + 2 MoE): MLA
# adapters alongside DSA cross-layer index sharing, TP=EP=4. Covers prepare
# (download + single-rank torch_dist conversion — any PP split of the toy
# starts a stage on a DSA skip layer) and the rollout -> train -> adapter-sync
# loop. The logprobs checker is disabled like the other pruned-toy CIs: the
# toy's rollout entropy legitimately exceeds the checker's real-model bound.

register_cuda_ci(est_time=2400, suite="stage-c-4-gpu-h200", labels=["lora-native"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.2_5layer",
        model_dir="/root/models",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=2,
        enable_wandb=False,
        extra_args="--ci-test --ci-disable-logprobs-checker ",
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
