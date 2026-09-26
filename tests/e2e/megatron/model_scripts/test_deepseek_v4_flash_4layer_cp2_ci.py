"""4-layer DeepSeek-V4-Flash RL on four GPUs with context parallelism (miles impl).

The base 4-layer case (test_deepseek_v4_flash_4layer_ci.py) at TP2 with sequence parallelism, CP2
with the all-gather CP split, EP4. Each micro-batch holds one unpacked sample, so the prune's CSA
layer always takes the load-balanced indexer path (each CP rank scores the paired row chunks and
returns the top-k picks), and the window, CSA and HCA layers all all-gather their KV across CP. The
engine runs without CP, so the train-rollout log-prob and KL gates check the whole CP path against it.
"""

import dataclasses
import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.model_scripts import test_deepseek_v4_flash_4layer_ci as base

register_cuda_ci(
    est_time=1900, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts"], hardware=["hopper", "blackwell"]
)

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
