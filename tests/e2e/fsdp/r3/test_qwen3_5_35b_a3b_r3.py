"""FSDP R3 on qwen3_5_moe (Qwen3.5-35B-A3B): 40 MoE layers, 256 experts, topk 8."""

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.fsdp.r3._common import CaseConfig, main

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h200", labels=["fsdp", "replay"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    model_name="Qwen3.5-35B-A3B",
    hf_repo="Qwen/Qwen3.5-35B-A3B",
    num_gpus=8,
    rollout_num_gpus_per_engine=2,
)


if __name__ == "__main__":
    main(CASE, wandb_file=__file__)
