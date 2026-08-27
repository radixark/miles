"""AMD 4-GPU variant of test_r3_mtp.py.

Standalone rather than an IS_HIP branch in the original: the MI300X fleet is
split into two 4-GPU runners, so the 8-GPU CUDA case cannot run there as
written, and keeping the variant separate means neither side's parallelism
constrains the other.

Difference from the CUDA case: cp_size 2 -> 1 and ep_size 4 -> 2, halving the
world size from 8 to 4. TP=2 and PP=2 are unchanged, so the MTP layer placement
under test is unaffected. Both axes have to shrink because Megatron sizes the
dense and expert grids independently -- world_size % (tp * cp * pp) == 0 and
world_size % (etp * ep * pp) == 0 -- so dropping CP alone leaves the expert grid
needing 8 ranks on a 4-rank world. CP and EP are the safe axes here: CP exists
for long-context throughput and EP for expert capacity, neither of which the
R3 + MTP behaviour under test depends on.
"""

import os

from tests.ci.ci_register import register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

register_rocm_ci(
    est_time=1100,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "amd"],
    disabled="FIXME: re-enable once this case passes on the MI350 runners.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")
register_ci_gate(
    metric_key="ci/r3_mismatch_fraction",
    constraint={"rel_up": 1.0, "abs_floor_up": 0.05, "rel_down": 1.0},
)

CASE = CaseConfig(
    use_deepep=False,
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=2,
    tp_size=2,
    ep_size=2,
    # GLM-4.7-Flash has 20 attention heads; non-EP SGLang TP must divide it.
    rollout_num_gpus_per_engine=4,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
