"""FSDP R3 on qwen3_moe (Qwen3-30B-A3B): 48 MoE layers, 128 experts, topk 8."""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.fsdp.r3._common import CaseConfig, main

register_cuda_ci(est_time=700, suite="stage-c-8-gpu-h200", labels=["fsdp", "replay"], hardware=["hopper"], num_gpus=8)
register_rocm_ci(est_time=600, suite="nightly-stage-c-8-gpu-mi350", labels=["fsdp", "replay"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="rollout/raw_reward")
register_ci_gate(metric_key="perf/actor_train_mfu")

CASE = CaseConfig(
    model_name="Qwen3-30B-A3B",
    hf_repo="Qwen/Qwen3-30B-A3B",
    num_gpus=8,
    rollout_num_gpus_per_engine=2,
)


if __name__ == "__main__":
    if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm":
        CASE.sglang_attention_backend = "triton"
        CASE.attn_implementation = "flash_attention_2"
    main(CASE, wandb_file=__file__)
