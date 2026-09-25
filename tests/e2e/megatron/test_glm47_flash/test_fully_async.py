import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "weight-update", "fully-async", "replay"],
    hardware=["hopper", "blackwell"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

# Fully-async rollout on the test_disagg_broadcast trainer (CP2, EP2, same token budget) with R3,
# MTP training and EAGLE on; two TP1 engines make the broadcast span more than one engine.
# in_place pause: retract-mode weight updates with R3 have known SGLang issues (arguments.py).
CASE = CaseConfig(
    use_deepep=False,
    num_gpus_per_node=2,
    cp_size=2,
    pp_size=1,
    tp_size=1,
    ep_size=2,
    colocate=False,
    rollout_num_gpus=2,
    rollout_num_gpus_per_engine=1,
    update_weight_transfer_mode="broadcast",
    num_rollout=2,
    fully_async=True,
    max_tokens_per_gpu=2048,
    rollout_max_response_len=4096,
    extra_args="--pause-generation-mode in_place ",
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
