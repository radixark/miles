"""AMD 4-GPU variant of test_mtp1_spec_v2_r3.py.

Qwen3.5-35B-A3B: 1 MTP layer + speculative-v2 + R3, on 4 GPUs.

Standalone rather than an IS_HIP branch in the original: the MI300X fleet is
split into two 4-GPU runners, so the 8-GPU CUDA case cannot run there as
written, and keeping the variant separate means neither side's parallelism
constrains the other.

Difference from the CUDA case: num_gpus_per_node 8 -> 4, which drops data
parallelism from 2 to 1. The tp2/pp2/cp1 shape is kept exactly -- TP=4 hits a
Qwen3.5 attention-output-gate sharding bug, CP=1 avoids the memory-heavy
GatedDeltaNet CP backward kernel, and PP=2 halves the resident layers. Because
TP/PP/CP are unchanged, the per-rank shard is identical to the CUDA case; the
original comment notes it fits 8x80GB, and MI300X carries 192GB per GPU, so 4
of them hold more total HBM than the node this was tuned for. The rollout
engine and SGLang EP follow the world size down from 8 to 4.
"""

import os

from tests.ci.ci_register import register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B_mtp._common import CaseConfig, execute, prepare

register_rocm_ci(
    est_time=1600,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "qwen35", "amd"],
    disabled="FIXME: re-enable once this case passes on the MI350 runners.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=2,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=4,
    sglang_ep_size=4,
    enable_mtp_training=True,
    use_r3=True,
    extra_args=("--moe-token-dispatcher-type alltoall " "--sglang-disable-shared-experts-fusion "),
    # miles has no VLM/vision implementation on the training side, so vision weights are
    # never synced; exclude them from the weight-equality check.
    check_weight_update_skip_list=("visual",),
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
