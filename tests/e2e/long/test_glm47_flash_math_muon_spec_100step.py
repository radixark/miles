"""GLM-4.7-Flash: 100-step DAPO-math long run with every advanced training feature on.

The long-run counterpart of tests/e2e/megatron/test_glm47_flash/test_r3_mtp.py: the same
DAPO-math-17k job and tp2/cp2/ep4/pp2 training topology, but 100 rollout steps, the
layer-wise distributed Muon optimizer (``--optimizer dist_muon``), single-GPU rollout engines,
EAGLE speculative decoding with MTP training, and R3 routing replay.

Single-GPU engines need the 141 GB H200: the 30B bf16 weights are ~60 GB, which leaves no KV
cache room on an 80 GB H100 (the H100 Qwen3.5 long run uses TP=2 for that reason).

Muon specifics:
- The Adam CPU-offload / precision-aware / fp16-state flags of the regular case require
  Megatron's distributed optimizer, which miles enables only for Adam; the Muon block offloads
  optimizer state between steps with ``--chunked-optimizer-state-offload`` instead.
- lr 1e-6, the Adam CI runs' value. The first attempt used 1e-5 (the 2-step Muon smoke test's
  value) and collapsed: reward 0.45 -> 0.69 at rollout 10, then a monotone slide to 0 by rollout
  60 while responses grew from 3.5k to 7.2k tokens (run 35719130821); train/rollout |dlogp| stayed
  at 0.001-0.005 throughout, so the drift was optimisation, not weight sync.

The per-step weight-equality check is off: with 100 updates it would add a full-model compare
to every step; the per-step train/rollout log-prob checker still guards weight-sync drift.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

# 100 steps must finish inside the workflow's 6 h job budget; 1.25x this estimate is the
# per-file timeout (see tests/ci/file_run.py), kept just under that budget so the run ends
# with a clean TERM and flushed logs rather than a hard cancel.
register_cuda_ci(est_time=17000, suite="stage-c-8-gpu-h200", labels=["long"], hardware=["hopper"])

MUON_OPTIMIZER_ARGS = (
    "--optimizer dist_muon "
    "--lr 1e-6 "
    "--lr-decay-style constant "
    "--weight-decay 0.1 "
    "--adam-beta1 0.9 "
    "--adam-beta2 0.98 "
    "--chunked-optimizer-state-offload "
    "--optimizer-state-offload-fraction 1.0 "
)

CASE = CaseConfig(
    use_deepep=False,
    num_gpus_per_node=8,
    cp_size=2,
    pp_size=2,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=1,
    num_rollout=100,
    optimizer_args=MUON_OPTIMIZER_ARGS,
    extra_args="--ci-disable-weight-update-checker ",
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
