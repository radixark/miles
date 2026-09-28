"""Qwen3.5-35B-A3B without speculative decoding or R3: the plain BF16 precision case.

The rollout runs DP attention (attention TP2 x DP2) with EP4 MoE and no DeepEP on either side.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500, suite="stage-c-4-gpu-h200", labels=["megatron", "qwen35"], hardware=["hopper", "blackwell"]
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    # tp2/pp2/ep2 (no CP) on 4x H200: the only 4-GPU Qwen3.5 case with PP.
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=2,
    tp_size=2,
    ep_size=2,
    megatron_dispatcher="alltoall",
    # One TP4 engine: attention TP2 x DP2, EP4.
    rollout_num_gpus_per_engine=4,
    sglang_dp_size=2,
    sglang_enable_dp_attention=True,
    sglang_ep_size=4,
    use_spec=False,
    enable_mtp_training=False,
    use_r3=False,
    check_weight_update_selector="target",
    # miles has no VLM/vision implementation on the training side, so vision weights are
    # never synced; exclude them from the weight-equality check.
    check_weight_update_skip_list=("visual",),
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
