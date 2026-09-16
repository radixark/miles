"""AMD 4-GPU variant of test_deepep_fp8_bridge.py.

Standalone rather than an IS_HIP branch in the original: the MI300X fleet is
split into two 4-GPU runners, so the 8-GPU CUDA case cannot run there as
written, and keeping the variant separate means neither side's parallelism
constrains the other.

Two differences from the CUDA case. The parallelism drops to the 4-GPU shape
already proven for this model in test_r3_baseline.py / test_r3_deepep_fp8.py
(cp2/pp1/tp2/ep4), and the rollout engine and SGLang EP follow the world size
down from 8 to 4. And the dispatchers differ: the training side uses the
alltoall token dispatcher, the rollout side uses mori. What the case covers is
otherwise unchanged: expert weights stay EP-sharded, so the bridge conversion
under FP8 rollout is exercised exactly as on CUDA. max_tokens_per_gpu is left
at the CUDA case's 2048 -- that cap was set by host memory, which the split
runner does not change.
"""

import os

from tests.ci.ci_register import register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_rocm_ci(
    est_time=800,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "amd"],
    disabled="FIXME: re-enable once this case passes on the MI350 runners.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=False,
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
    extra_args=("--sglang-moe-a2a-backend mori " "--sglang-deepep-mode auto "),
    extra_env_vars={"SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16384"},
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
