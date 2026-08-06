import os

from examples.lora.run_lora_native import ScriptArgs, _prepare, _train
from tests.ci.ci_register import register_cuda_ci

# Native (raw-mode) LoRA on GLM-4.7-Flash: the MLA registry's reference model —
# replicated q_a/kv_a down-projections plus column-parallel up-projections
# under TP2/EP4. Covers prepare (download + torch_dist conversion) and the full
# rollout -> train -> adapter-sync loop with the CI checkers on (cross-engine
# logprob agreement at abs_tol 0.03, step-0 ppo_kl).

register_cuda_ci(est_time=3600, suite="stage-c-8-gpu-h200", labels=["lora-native"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-4.7-Flash",
        model_dir="/root/models",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        enable_wandb=False,
        extra_args="--ci-test ",
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
