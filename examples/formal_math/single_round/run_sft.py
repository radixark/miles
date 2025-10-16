import datetime
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "tests"))

import command_utils as U

dataset_transform_id = os.environ["MILES_DATASET_TRANSFORM_ID"]

MODEL_NAME, MODEL_TYPE = "Qwen3-8B-Base", "qwen3-8B"

NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.convert_checkpoint(model_name=MODEL_NAME, model_type=MODEL_TYPE, num_gpus=NUM_GPUS)


def execute():
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"

    CKPT_ARGS = (
        "--hf-checkpoint /root/Qwen3-4B-Base/ "
        "--ref-load /root/Qwen3-4B-Base_torch_dist "
        "--load /root/Qwen3-4B-Base_miles/ "
        "--save /root/Qwen3-4B-Base_miles/ "
        "--save-interval 1000 "
    )

    SFT_ARGS = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        "--prompt-data /root/openhermes2_5.parquet "
        "--input-key messages "
        "--rollout-shuffle "
        "--num-epoch 3 "
        "--rollout-batch-size 128 "
        "--global-batch-size 128 "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        "--debug-train-only "
    )

    PERF_ARGS = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "

        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "

        # --micro-batch-size 1
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    OPTIMIZER_ARGS = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-warmup-iters 128 "
        "--lr-decay-style cosine "
        "--min-lr 1e-6 "
        "--lr-warmup-fraction 0.9 "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.95 "
    )

    WANDB_ARGS = (
        # --use-wandb
        # --wandb-project miles-dev
        # --wandb-group qwen3-4B-base-sft
        # --wandb-key ${WANDB_KEY}
    )

    MISC_ARGS = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=NUM_GPUS,
        model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    execute()
