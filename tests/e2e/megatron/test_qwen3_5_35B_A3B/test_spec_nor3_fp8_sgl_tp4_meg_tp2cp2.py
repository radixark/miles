"""Qwen3.5-35B-A3B: 0 MTP layers + speculative-v2 (no R3) on an FP8 TP-only rollout.

MTP training is OFF and R3 is off. The rollout still runs EAGLE spec from the
checkpoint draft, whose MTP weights are never synced from training, so this case
sets the weight-check selector to "target": only the target (main) model is
checked, skipping the draft.

The rollout is one TP4 engine without EP or DP attention, so SGLang shards every expert by
TP. It serves Qwen/Qwen3.5-35B-A3B-FP8 against bf16 training, so every weight update
re-quantizes and the equality check allows quantization error.

Vision weights are also excluded (--check-weight-update-skip-list visual): miles has no
VLM/vision implementation on the training side, so they are never synced.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B._common import CaseConfig, execute, prepare

# Hopper only until a B200 run of this case passes: on Blackwell, FP8 without DeepEP pairs UE8M0
# linear scales with fp32 expert scales (quantizer_fp8.py).
register_cuda_ci(est_time=2100, suite="stage-c-4-gpu-h200", labels=["megatron", "qwen35"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    # tp2/cp2/ep4: TP=4 hits a Qwen3.5 attention-output-gate sharding bug, so stay at TP=2.
    # PP=1 on 4 GPUs: PP=2 stays covered by test_nospec_nor3_bf16_sgl_dpattn2x2_meg_tp2pp2.
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    # 4096 (the CP=1 cases keep 8192): CP=2 routes the GatedDeltaNet backward through the heavier
    # fla CP kernel, whose Triton autotune OOMs at 8192 even with PP=2; halve the budget for headroom.
    max_tokens_per_gpu=4096,
    # One TP4 engine, no SGLang EP: every FP8 projection shard stays 128x128-block aligned at TP4
    # (MoE and shared-expert intermediate 512 / 4 = 128).
    rollout_num_gpus_per_engine=4,
    use_fp8_rollout=True,
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
