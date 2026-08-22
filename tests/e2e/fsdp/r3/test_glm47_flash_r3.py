"""FSDP R3 on glm4_moe_lite (GLM-4.7-Flash): 47 decoder layers of which 46 are MoE
(``first_k_dense_replace=1`` makes layer 0 dense), 64 experts, topk 4, group-limited routing.

This is the case that exercises keying replay streams by global layer index: an ordinal over
MoE blocks would shift every layer by one against the rollout tensor.
"""

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.fsdp.r3._common import CaseConfig, main

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h200", labels=["fsdp", "replay"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="rollout/raw_reward")
register_ci_gate(metric_key="ci/r3_mismatch_fraction")

CASE = CaseConfig(
    model_name="GLM-4.7-Flash",
    hf_repo="zai-org/GLM-4.7-Flash",
    num_gpus=8,
    rollout_num_gpus_per_engine=2,
)


if __name__ == "__main__":
    main(CASE, wandb_file=__file__)
