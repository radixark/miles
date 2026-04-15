"""P2P weight transfer test for Qwen3-235B-A22B across 16 nodes (128 GPUs).

Uses the big-run config from docs/bigrun/parallelism-breakdown.md:
  Trainer: 32 GPUs (4 nodes), TP=4, PP=4, CP=2, EP=8
  Sampler: 96 GPUs (12 nodes), 12 engines × 8 GPUs, TP=8, EP=2, DP=2
"""

import os
import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-235B-A22B"
MODEL_TYPE = "qwen3-235B-A22B"
NUM_GPUS_PER_NODE = 8

HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "/data/models/Qwen3-235B-A22B")
TORCH_DIST_CHECKPOINT = os.environ.get("TORCH_DIST_CHECKPOINT", "/data/models/Qwen3-235B-A22B_torch_dist")
PROMPT_DATA = os.environ.get("PROMPT_DATA", "/data/dapo-math-17k/dapo-math-17k.jsonl")

# Trainer: 4 nodes × 8 GPUs = 32 GPUs
ACTOR_NUM_NODES = 4
ACTOR_NUM_GPUS_PER_NODE = 8

# Sampler: 12 engines × 8 GPUs = 96 GPUs
ROLLOUT_NUM_GPUS_PER_ENGINE = 8
ROLLOUT_NUM_GPUS = 96


def execute():
    ckpt_args = f"--hf-checkpoint {HF_CHECKPOINT} " f"--ref-load {TORCH_DIST_CHECKPOINT} "

    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 22 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    # Per-GPU RDMA NIC mapping for B200 nodes (GPU index -> PIX-affine mlx5 device)
    ib_device_json = (
        '{"0":"mlx5_1","1":"mlx5_5","2":"mlx5_2","3":"mlx5_0","4":"mlx5_11","5":"mlx5_15","6":"mlx5_14","7":"mlx5_6"}'
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {ROLLOUT_NUM_GPUS_PER_ENGINE} "
        f"--rollout-num-gpus {ROLLOUT_NUM_GPUS} "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 2 "
        "--sglang-ep-size 2 "
        "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
        f"--update-weight-p2p-ib-device '{ib_device_json}' "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {ACTOR_NUM_NODES} "
        f"--actor-num-gpus-per-node {ACTOR_NUM_GPUS_PER_NODE} "
        f"--update-weight-buffer-size {4 * 1024 ** 3} "
        "--update-weight-transfer-mode p2p "
        "--check-weight-update-equal "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS_PER_NODE,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    execute()
