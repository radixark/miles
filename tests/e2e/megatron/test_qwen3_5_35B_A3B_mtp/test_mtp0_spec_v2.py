"""Qwen3.5-35B-A3B: 0 MTP layers + speculative-v2 (no R3).

MTP training is OFF and R3 is off. The rollout still runs EAGLE spec from the
checkpoint draft, whose MTP weights are never synced from training. Whether that
draft is checked is governed by the weight-check selector, not the skip list.

The suite-wide vision skip (--check-weight-update-skip-list visual, applied in
_common) covers Qwen3.5's vision tower, which text RL never updates.

TODO(selector): MTP-weight exclusion for this 0-layer case should be driven by
--check-weight-update-selector once it gains an MTP-excluding mode; the selector
currently only supports "all".
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.test_qwen3_5_35B_A3B_mtp._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1200, suite="stage-c-8-gpu-h100", labels=["megatron", "qwen35"])

CASE = CaseConfig(
    num_gpus_per_node=8,
    cp_size=2,
    pp_size=1,
    tp_size=2,
    ep_size=8,
    rollout_num_gpus_per_engine=8,
    sglang_ep_size=8,
    enable_mtp_training=False,
    use_r3=False,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
