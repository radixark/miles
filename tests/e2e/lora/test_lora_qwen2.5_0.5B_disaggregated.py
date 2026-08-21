"""E2E test for LoRA training with Qwen2.5-0.5B on GSM8K, disaggregated.

The colocate sibling (test_lora_qwen2.5_0.5B.py) covers the CUDA/HIP-IPC weight-sync path.
This one covers the other transport: actor and rollout engines on disjoint GPUs, adapters
pushed over the NCCL/RCCL broadcast group via UpdateWeightFromDistributed.  That path has its
own base/adapter sync logic, so a colocate-only suite leaves it unguarded.

Validates:
  - LoRA model setup via Bridge
  - base-weight sync under --check-weight-update-equal, which --ci-test turns on
  - a rollout after the first optimizer update
  - Training completes without errors

Requires: 4 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import os

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=400, suite="stage-c-4-gpu-h200", labels=["lora"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4
ACTOR_GPUS = 2
ROLLOUT_GPUS = NUM_GPUS - ACTOR_GPUS


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " "--megatron-to-hf-mode bridge "

    lora_args = "--lora-rank 32 " "--lora-alpha 32 " "--lora-dropout 0.0 " '--target-modules "all-linear" '

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.4 "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--use-miles-router "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {ACTOR_GPUS} "
        f"--rollout-num-gpus {ROLLOUT_GPUS} "
        "--update-weight-transfer-mode broadcast "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
