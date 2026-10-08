"""Qwen3.5-35B-A3B: p2p weight updates into an EAGLE engine whose MTP draft is trained.

4 train GPUs (TP1, DP4, EP4) write both the target and its MTP draft straight into one TP4 engine (experts sharded
by TP, no DP attention, no DeepEP) over Mooncake; the equality check (selector "all") covers both models.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B._common import CaseConfig, execute, prepare

# 8x H200 because EP must stay >= 4 (see test_nospec_r3_bf16_sgl_dpattn1x4_meg_tp1cp1_fullyasync.py)
register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "qwen35", "weight-update"],
    hardware=["hopper"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=1,
    tp_size=1,
    ep_size=4,
    megatron_dispatcher="alltoall",
    colocate=False,
    rollout_num_gpus=4,
    rollout_num_gpus_per_engine=4,
    update_weight_transfer_mode="p2p",
    enable_mtp_training=True,
    use_r3=False,
    # miles has no VLM/vision implementation on the training side, so vision weights are
    # never synced; exclude them from the weight-equality check.
    check_weight_update_skip_list=("visual",),
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
