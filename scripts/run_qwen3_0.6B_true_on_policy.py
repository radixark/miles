import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))

import command_utils as U

MODEL_NAME = "Qwen3-0.6B"

MODE = os.environ.get("MILES_SCRIPT_MODE", "normal")
assert MODE in {"normal", "debug_one_sample"}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {1 if MODE == 'debug_one_sample' else 3000} "
        f"--rollout-batch-size {32 if MODE == 'debug_one_sample' else 32} "
        f"--n-samples-per-prompt {8 if MODE == 'debug_one_sample' else 8} "
        f"--rollout-max-response-len {10 if MODE == 'debug_one_sample' else 1024} "
        "--rollout-temperature 0.8 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {256 if MODE == 'debug_one_sample' else 256} "
    )

    eval_args = ""
    if MODE != "debug_one_sample":
        eval_args = (
            "--eval-interval 20 "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        # mainly to look at its metric
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-enable-deterministic-inference "
        "--sglang-attention-backend fa3 "
        f"--sglang-mem-fraction-static 0.35 "
        # f"{'--sglang-disable-cuda-graph ' if MODE == 'debug_one_sample' else ''}"
    )

    fsdp_args = (
        # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
        # "--fsdp-full-params "  # Uncomment this line to enable full params mode
        # Set the bucket size for weight update
        "--update-weight-buffer-size 536870912 "  # 512MB
        "--attn-implementation flash_attention_3 "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--colocate "
        "--train-backend fsdp "
        "--deterministic-mode "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=1,
        model_type=None,
        extra_env_vars={
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "SGLANG_DUMPER_ENABLE": "0",  # temporary
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
