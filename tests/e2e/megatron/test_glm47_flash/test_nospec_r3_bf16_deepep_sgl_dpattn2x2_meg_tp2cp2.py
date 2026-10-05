"""GLM-4.7-Flash with distributed Muon, R3 and DeepEP on both sides, without speculative decoding.

Colocated on 4 GPUs: Megatron trains at TP2 x CP2 with EP4 over DeepEP, and one BF16 SGLang engine
runs DP attention (attention TP2 x DP2) with EP4 and DeepEP auto (normal dispatch for prefill,
low-latency for decode) on the DeepGEMM runner. Dispatch is BF16, not the FP8 that "auto" picks.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

# 4x H200: the 8x H200 runners cannot bring NVSHMEM up over IB, which DeepEP low-latency mode needs.
register_cuda_ci(est_time=1500, suite="stage-c-4-gpu-h200", labels=["megatron", "replay"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    optimizer="dist_muon",
    use_deepep=True,
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=4,
    # Attention TP2 x DP2; GLM-4.7-Flash's 20 attention heads split over the attention TP.
    sglang_dp_size=2,
    sglang_enable_dp_attention=True,
    # SGLang DeepEP forces EP to the engine TP.
    sglang_ep_size=4,
    sglang_deepep_dispatcher_output_dtype="bf16",
    use_spec=False,
    use_r3=True,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
