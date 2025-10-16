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

    train_args = TODO

    U.execute_train(
        train_args=train_args,
        num_gpus=NUM_GPUS,
        model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    execute()
