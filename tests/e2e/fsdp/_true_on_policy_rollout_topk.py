"""Shared launch of the Qwen3-4B FSDP true-on-policy GRPO runs that record rollout top-k candidates.

Under --ci-test the forward-only pass scores SGLang's recorded candidates (``Sample.rollout_topk_token_ids`` /
``rollout_topk_log_probs``) with the trainer's true-on-policy log-probs, and ``log_rollout_data`` asserts on every
rollout that ``rollout/topk_logprob_abs_diff`` and ``rollout/topk_set_mismatch`` are exactly 0, next to its
``log_probs == rollout_log_probs`` check. The sampler is unfiltered, so the candidates are the full-vocabulary top-k.
"""

import os

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
NUM_GPUS = 2
TOP_LOGPROBS_NUM = 128  # the router's limit


def prepare() -> None:
    backend = U.default_config().create_backend()
    backend.exec_command_cpu("mkdir -p /root/models /root/datasets")
    backend.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    backend.hf_download_dataset("zhuzilin/gsm8k")


def execute(test_file: str, rollout_path_args: str) -> None:
    """Run plain GRPO; ``rollout_path_args`` selects how rollouts reach SGLang."""
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        # Short answers that finish: the session server's TITO check under --ci-test fails on truncated turns.
        """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1 "
        f"--rollout-top-logprobs-num {TOP_LOGPROBS_NUM} "
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

    grpo_args = "--advantage-estimator grpo " "--kl-loss-coef 0.00 " "--kl-coef 0.00 " "--entropy-coef 0.00 "

    fsdp_args = "--train-backend fsdp " "--update-weight-buffer-size 536870912 "

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-decode-log-interval 1000 "

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
        + rollout_path_args
        + optimizer_args
        + grpo_args
        + fsdp_args
        + sglang_args
        + true_on_policy_args
        + U.get_default_wandb_args(test_file)
        + "--ci-test --no-enable-sample-ownership-checker "
        + f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
    )

    backend = U.default_config().create_backend()
    backend.execute_train(
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


def main(test_file: str, rollout_path_args: str) -> None:
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(test_file, rollout_path_args)
