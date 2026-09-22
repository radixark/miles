"""Qwen3.5-35B-A3B: 100-step DAPO-math long run with every advanced training feature on.

The long-run counterpart of tests/e2e/megatron/test_qwen3_5_35B_A3B_mtp/test_mtp1_spec_v2_r3.py:
the same DAPO-math-17k job, but 100 rollout steps, the layer-wise distributed Muon optimizer
(``--optimizer dist_muon``), tp2/cp2/ep4/pp2 training parallelism, EAGLE speculative decoding
(spec-v2) with MTP training, and R3 routing replay.

Rollout engines are TP=2 rather than single-GPU: the 35B bf16 weights alone are ~70 GB, more
than one 80 GB H100 can hold next to any KV cache (the H200 GLM-4.7-Flash long run uses TP=1).

Muon specifics:
- ``--rematerialize-param-from-master-weight`` and the Adam CPU-offload / precision-aware flags
  require Megatron's distributed optimizer, which miles enables only for Adam; the Muon block
  offloads optimizer state between steps with ``--chunked-optimizer-state-offload`` instead.
- lr 1e-5 follows tests/e2e/megatron/test_qwen3_4B_muon_offload_disk.py.

The per-step weight-equality check is off: with 100 updates it would add a full-model compare
to every step; the per-step train/rollout log-prob checker still guards weight-sync drift.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.test_qwen3_5_35B_A3B_mtp._common import CaseConfig, execute, prepare

# 100 steps must finish inside the workflow's 6 h job budget; 1.25x this estimate is the
# per-file timeout (see tests/ci/file_run.py), kept just under that budget so the run ends
# with a clean TERM and flushed logs rather than a hard cancel.
register_cuda_ci(est_time=17000, suite="stage-c-8-gpu-h100", labels=["long", "qwen35"], hardware=["hopper"])

MUON_OPTIMIZER_ARGS = (
    "--optimizer dist_muon "
    "--lr 1e-5 "
    "--lr-decay-style constant "
    "--weight-decay 0.1 "
    "--adam-beta1 0.9 "
    "--adam-beta2 0.98 "
    "--chunked-optimizer-state-offload "
    "--optimizer-state-offload-fraction 1.0 "
)

CASE = CaseConfig(
    num_gpus_per_node=8,
    cp_size=2,
    pp_size=2,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=2,
    sglang_ep_size=2,
    enable_mtp_training=True,
    use_r3=True,
    # 4096, not the regular 8192: with cp2 the GatedDeltaNet CP backward (fla chunk_delta_h)
    # OOMed on 8x80GB next to ~41 GB of resident Muon training state (run 35719111404);
    # halving the tokens per micro-batch halves that activation working set.
    max_tokens_per_gpu=4096,
    num_rollout=100,
    optimizer_args=MUON_OPTIMIZER_ARGS,
    rematerialize_param_from_master_weight=False,
    # miles has no VLM/vision implementation on the training side; kept for parity with the
    # regular case even though the per-step weight check is disabled below.
    check_weight_update_skip_list=("visual",),
    extra_args="--ci-disable-weight-update-checker ",
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
