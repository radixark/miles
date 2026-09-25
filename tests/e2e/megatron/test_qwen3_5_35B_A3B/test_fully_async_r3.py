"""Qwen3.5-35B-A3B fully-async rollout with R3 and without speculative decoding.

2 train GPUs (TP1, DP2, EP2) + one 2-GPU engine with DP attention (attention TP1 x DP2, EP2),
no DeepEP on either side. Weights reach the engine by broadcast.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "qwen35", "weight-update", "fully-async", "replay"],
    hardware=["hopper", "blackwell"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    num_gpus_per_node=2,
    cp_size=1,
    pp_size=1,
    tp_size=1,
    ep_size=2,
    megatron_dispatcher="alltoall",
    colocate=False,
    rollout_num_gpus=2,
    rollout_num_gpus_per_engine=2,
    sglang_dp_size=2,
    sglang_enable_dp_attention=True,
    sglang_ep_size=2,
    fully_async=True,
    use_spec=False,
    enable_mtp_training=False,
    use_r3=True,
    check_weight_update_selector="target",
    # miles has no VLM/vision implementation on the training side, so vision weights are
    # never synced; exclude them from the weight-equality check.
    check_weight_update_skip_list=("visual",),
    # The 2-GPU trainer holds half the experts plus every dense weight per GPU; the GLM-4.7-Flash
    # 2+2 token budget keeps activations small.
    max_tokens_per_gpu=2048,
    rollout_max_response_len=4096,
    # in_place pause, as in the GLM-4.7-Flash fully-async R3 case.
    extra_args="--pause-generation-mode in_place ",
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
