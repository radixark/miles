"""AMD 4-GPU Qwen3.5-35B-A3B case: 1 MTP layer + speculative-v2 + R3.

MTP training is on with one draft layer, so the rollout MTP/draft weights are synced
from training (selector "all" checks target and draft; vision weights are skipped).
Training runs TP2 x PP2 with EP2, which keeps PP2 * EP2 * expert-TP1 equal to world
size 4; the rollout is one TP4 engine with EP4.

A standalone AMD file rather than an IS_HIP branch in a CUDA case: the AMD fleet runs
4-GPU runners, and keeping the file separate means neither side's parallelism
constrains the other.
"""

import os

from tests.ci.ci_register import register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B._common import CaseConfig, execute, prepare

register_rocm_ci(
    est_time=1600,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "qwen35", "amd"],
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
    ep_size=2,
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
