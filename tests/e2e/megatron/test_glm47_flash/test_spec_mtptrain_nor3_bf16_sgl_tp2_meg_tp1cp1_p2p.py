"""GLM-4.7-Flash: verify target and trained MTP draft equality after P2P updates without R3."""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "weight-update"],
    hardware=["hopper"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=False,
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=1,
    tp_size=1,
    ep_size=4,
    colocate=False,
    rollout_num_gpus=4,
    rollout_num_gpus_per_engine=2,
    use_spec=True,
    use_r3=False,
    update_weight_transfer_mode="p2p",
    num_rollout=2,
    # CP1 must fit the full prompt and response on one GPU
    max_tokens_per_gpu=8192,
    rollout_max_response_len=4096,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
