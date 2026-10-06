"""Smoke run of DeepSeek-V4-Flash's CP path: the 4-layer RL case on four GPUs (miles impl).

The base case (test_deepseek_v4_flash_4layer_ci.py) at TP2 with sequence parallelism, CP2 with the
all-gather CP split, EP4. Each micro-batch holds one unpacked sample, so the CSA layer always takes
the load-balanced indexer path. The run catches crashes, hangs and train-side nondeterminism, not
wrong picks: its train-rollout metrics are tracked but not gated, and short GSM8K samples have fewer
compressed keys than the top-k keeps. tests/fast-gpu/test_dsv4_indexer_cp_balance.py checks the picks.
"""

import dataclasses
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.model_scripts import test_deepseek_v4_flash_4layer_ci as base

register_cuda_ci(
    est_time=1900, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts"], hardware=["hopper", "blackwell"]
)
register_rocm_ci(est_time=700, suite="nightly-stage-c-4-gpu-mi350", labels=["megatron", "model-scripts"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

prepare = base.prepare
execute = base.execute


def _args():
    return dataclasses.replace(base._args(), cp_size=2)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
