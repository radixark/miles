import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=600,
    suite="stage-c-2-gpu-h200",
    labels=["fsdp", "sglang", "replay"],
    hardware=["hopper"],
)
# The log-prob diff and both KLs stay at 0 under true-on-policy; a bf16 flip (see execute) adds only about 2e-7 to
# the diff. ppo_kl compares the training forward with forward-only scoring, so it also covers the loss-path mask.
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="train/ppo_kl")

MODEL_NAME = "Qwen3-0.6B"
NUM_GPUS = 2


def prepare() -> None:
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute() -> None:
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    # Top-p/top-k enables sampling-support replay; --ci-test asserts log_probs == rollout_log_probs on every rollout.
    # Equality holds only through bf16 rounding: SGLang's fp32 log(p / sum_S p) and the trainer's masked bf16
    # log_softmax differ by one ulp on ~4.6e-6 of sampled tokens, which rarely moves the bf16 per-sample sums checked.
    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rm-type math "
        "--num-rollout 5 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--rollout-top-p 0.8 "
        "--rollout-top-k 32 "
        "--global-batch-size 16 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    algorithm_args = "--advantage-estimator grpo " "--kl-loss-coef 0.00 " "--kl-coef 0.00 " "--entropy-coef 0.00 "

    fsdp_args = "--train-backend fsdp " "--update-weight-buffer-size 536870912 "

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.4 " "--sglang-decode-log-interval 1000 "
    )

    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1 "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
        "--deterministic-mode "
        "--true-on-policy-mode "
    )

    train_args = (
        ckpt_args
        + rollout_args
        + optimizer_args
        + algorithm_args
        + fsdp_args
        + sglang_args
        + true_on_policy_args
        + U.get_default_wandb_args(__file__)
        + "--ci-test --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars={
            "NCCL_ALGO": "allreduce:tree",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
