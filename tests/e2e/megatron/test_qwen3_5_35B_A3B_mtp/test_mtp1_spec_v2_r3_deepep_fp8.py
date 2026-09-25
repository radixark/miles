"""Qwen3.5-35B-A3B: 1 MTP layer + speculative-v2 + R3 + DeepEP on both sides + FP8 rollout.

MTP training and R3 are on as in test_mtp1_spec_v2_r3 (selector "all" checks target and draft;
vision weights are skipped). SGLang also runs DeepEP (normal mode for prefill/extend batches,
low-latency for every decode-phase forward) and serves Qwen/Qwen3.5-35B-A3B-FP8 against bf16
training, so every weight update re-quantizes and the equality check allows quantization error.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B_mtp._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=2200, suite="stage-c-4-gpu-h200", labels=["megatron", "qwen35", "replay"], hardware=["hopper"]
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    # tp2/pp1/cp1/ep4: dense DP2 with EP4 folded over TP x DP, the only 4-GPU Qwen3.5 case
    # with dense DP > 1. TP=4 hits the Qwen3.5 attention-output-gate sharding bug.
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=4,
    sglang_ep_size=4,
    enable_mtp_training=True,
    use_r3=True,
    use_deepep=True,
    use_fp8_rollout=True,
    # miles has no VLM/vision implementation on the training side, so vision weights are
    # never synced; exclude them from the weight-equality check.
    check_weight_update_skip_list=("visual",),
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
