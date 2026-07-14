"""Qwen3.5-35B-A3B: 1 MTP layer + speculative-v2 + R3.

MTP training is on with one draft layer, so the rollout MTP/draft weights are synced
from training (selector "all" checks both target and draft). Vision weights are still
excluded: miles has no VLM/vision implementation on the training side.
"""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.megatron.test_qwen3_5_35B_A3B_mtp._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1200, suite="stage-c-8-gpu-h100", labels=["megatron", "qwen35"])
register_rocm_ci(est_time=1200, suite="stage-c-8-gpu-mi300x", labels=["megatron", "qwen35"])

CASE = CaseConfig(
    # tp2/pp2/cp1/ep4: TP=4 hits a Qwen3.5 attention-output-gate sharding bug, so stay at
    # TP=2; CP=1 avoids the memory-heavy GatedDeltaNet CP backward kernel and PP=2 halves
    # the resident layers, together fitting the MTP-training run on 8x80GB.
    num_gpus_per_node=8,
    cp_size=1,
    pp_size=2,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=8,
    sglang_ep_size=8,
    enable_mtp_training=True,
    use_r3=True,
    # miles has no VLM/vision implementation on the training side, so vision weights are
    # never synced; exclude them from the weight-equality check.
    check_weight_update_skip_list=("visual",),
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
