"""AMD 4-GPU variant of test_deepep_fp8_bridge.py.

Standalone rather than an IS_HIP branch in the original: the MI300X fleet is
split into two 4-GPU runners, so the 8-GPU CUDA case cannot run there as
written, and keeping the variant separate means neither side's parallelism
constrains the other.

Difference from the CUDA case: the parallelism drops to the 4-GPU shape already
proven for this model in test_r3_baseline.py / test_r3_deepep_fp8.py
(cp2/pp1/tp2/ep4), and the rollout engine and SGLang EP follow the world size
down from 8 to 4. The feature flags under test are unchanged: DeepEP on, FP8
rollout on, bridge on, R3 off. max_tokens_per_gpu is left at the CUDA case's
2048 -- that cap was set by host memory, which the split runner does not
change.
"""

import os

from tests.ci.ci_register import register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_rocm_ci(
    est_time=800,
    suite="stage-c-4-gpu-mi300x",
    labels=["megatron", "amd"],
    disabled="Disable due to failure",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=True,
    use_fp8_rollout=True,
    use_int4_rollout=False,
    use_bridge=True,
    use_r3=False,
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=4,
    sglang_ep_size=4,
    max_tokens_per_gpu=2048,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
