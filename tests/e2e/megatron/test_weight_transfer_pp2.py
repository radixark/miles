"""
Integration test for multi-PP concurrent weight transfer.

Verifies that the concurrent broadcast from all PP sources (PR #788)
correctly transfers weights from Megatron training actors to SGLang
rollout engines when pipeline-model-parallel-size > 1.

Usage (inside the miles Docker container, 8 GPUs):
    # Mount host model/dataset dirs to /root/models and /root/datasets:
    podman run --rm -it --device nvidia.com/gpu=all --shm-size=16g \
      -v /data/users/mogicianwu/github/miles:/root/miles \
      -v /data/users/mogicianwu/models:/root/models \
      -v /data/users/mogicianwu/datasets:/root/datasets \
      -w /root/miles radixark/miles:nightly-dev-20260113a bash

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tests/e2e/megatron/test_weight_transfer_pp2.py

The test:
  1. Downloads Qwen3-0.6B (small model, fast checkpoint conversion)
  2. Converts to Megatron distributed format with PP=2, TP=2 (4 train GPUs)
  3. Runs 1 rollout with weight transfer from all PP sources concurrently
  4. If the weight transfer is broken, the rollout/training step will fail.
"""

import os
from pathlib import Path

import miles.utils.external_utils.command_utils as U
from miles.utils.misc import exec_command

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 8

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[3]


def prepare():
    exec_command("mkdir -p /root/models /root/datasets")
    # If model/dataset are pre-downloaded and mounted, skip download.
    # Otherwise, download them (requires network access).
    if not os.path.isdir(f"/root/models/{MODEL_NAME}"):
        exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    if not os.path.isfile("/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"):
        U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}_torch_dist"

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32"
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048"
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28"
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98"
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        "--rollout-num-gpus 4 "
        "--sglang-mem-fraction-static 0.8"
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        f"--update-weight-buffer-size {1 * 1024 ** 3}"
    )

    all_args = f"{ckpt_args} {rollout_args} {perf_args} {grpo_args} {optimizer_args} {sglang_args} {misc_args}"

    U.execute_train(
        train_args=all_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
