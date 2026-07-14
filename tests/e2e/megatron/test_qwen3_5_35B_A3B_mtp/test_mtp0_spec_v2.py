"""Qwen3.5-35B-A3B: 0 MTP layers + speculative-v2 (no R3).

MTP training is OFF and R3 is off. The rollout still runs EAGLE spec from the
checkpoint draft, whose MTP weights are never synced from training, so this case
sets the weight-check selector to "target": only the target (main) model is
checked, skipping the draft.

Vision weights are also excluded (--check-weight-update-skip-list visual): miles has no
VLM/vision implementation on the training side, so they are never synced.
"""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.megatron.test_qwen3_5_35B_A3B_mtp._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1200, suite="stage-c-8-gpu-h100", labels=["megatron", "qwen35"])
register_rocm_ci(est_time=1200, suite="stage-c-8-gpu-mi300x", labels=["megatron", "qwen35"])

CASE = CaseConfig(
    # tp2/cp2/pp2/ep4: TP=4 hits a Qwen3.5 attention-output-gate sharding bug, so stay at
    # TP=2 and use PP=2 to halve the resident layers for OOM headroom (keeps CP=2 coverage).
    num_gpus_per_node=8,
    cp_size=2,
    pp_size=2,
    tp_size=2,
    ep_size=4,
    # 4096 (mtp1 keeps 8192): CP=2 routes the GatedDeltaNet backward through the heavier fla
    # CP kernel, whose Triton autotune OOMs at 8192 even with PP=2; halve the budget for headroom.
    max_tokens_per_gpu=4096,
    rollout_num_gpus_per_engine=8,
    sglang_ep_size=8,
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
