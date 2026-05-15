"""E2E test for multi-LoRA training with Qwen3-4B on GSM8K + DAPO-Math.

Spins up the multi-LoRA trainer with two CLI-registered adapters and runs until
both drain (via --multi-lora-disable-service-mode), validating:
  - Multi-LoRA controller register/drain lifecycle
  - Per-adapter slot allocation and weight sync to SGLang
  - Round-robin sampling across two adapters

Requires: 8 GPUs, Qwen3-4B model, gsm8k + dapo-math-17k datasets.
Triggered by label: run-ci-lora
"""

import os
from pathlib import Path

import yaml

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=1200, suite="stage-c-multi-lora-8-gpu", num_gpus=8)


MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8

ADAPTER_DIR = Path("/tmp/multi_lora_ci_adapters")

ADAPTER_CONFIGS = {
    "dapo_math": {
        "rank": 32,
        "alpha": 32,
        "data": "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl",
        "input_key": "prompt",
        "label_key": "label",
        "rm_type": "deepscaler",
        "num_row": 100,
    },
    "gsm8k": {
        "rank": 16,
        "alpha": 16,
        "data": "/root/datasets/gsm8k/train.parquet",
        "input_key": "messages",
        "label_key": "label",
        "rm_type": "math",
        "num_row": 80,
    },
}


def write_adapter_yamls():
    for name, cfg in ADAPTER_CONFIGS.items():
        d = ADAPTER_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter.yaml").write_text(yaml.safe_dump({**cfg, "dir": str(d)}))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/gsm8k")
    write_adapter_yamls()


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --megatron-to-hf-mode bridge "

    lora_args = (
        "--lora-rank 32 "
        "--lora-alpha 32 "
        "--lora-dropout 0.0 "
        '--target-modules "all-linear" '
        "--multi-lora-n-adapters 4 "
        "--multi-lora-idle-poll-s 5 "
        "--multi-lora-disable-service-mode "
        f'--multi-lora-adapter "dapo_math" "{ADAPTER_DIR}/dapo_math/adapter.yaml" '
        f'--multi-lora-adapter "gsm8k" "{ADAPTER_DIR}/gsm8k/adapter.yaml" '
        "--sglang-lora-backend triton "
    )

    rollout_args = (
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 50 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 64 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 "

    save_args = "--save-interval 5 --save /root/checkpoints/multi_lora-qwen3-4B-ci "

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
        f"--actor-num-gpus-per-node {NUM_GPUS} "
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
        f"{sglang_args} "
        f"{ci_args} "
        f"{save_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="examples/multi_lora/train_multi_lora.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
