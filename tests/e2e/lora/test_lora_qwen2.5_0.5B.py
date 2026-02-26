"""E2E test for LoRA training with Qwen2.5-0.5B on GSM8K.

Uses the Megatron backend with bridge mode.  Runs a short GRPO training loop
with LoRA enabled (rank=32, all-linear) to validate:
  - LoRA model setup via Bridge
  - LoRA weight sync to SGLang rollout engines
  - LoRA checkpoint save (native + HF PEFT format)
  - Training completes without errors

Requires: 8 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import os

import miles.utils.external_utils.command_utils as U

ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")


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
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    eval_args = (
        f"{'--eval-interval 2 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
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

    save_args = "--save-interval 2 " "--save /root/checkpoints/lora-qwen2.5-0.5B-ci "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--use-miles-router "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{save_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
