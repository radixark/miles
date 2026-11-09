"""The R3 the trainer consumes is the same through the output store as inline, on every DP and CP rank.

Two three-rollout runs differ only in --use-output-store: a 5-layer Qwen3-30B-A3B slice trained on six GPUs
(DP 2 x CP 3) and served by two one-GPU SGLang engines. Deterministic inference with fixed per-sample seeds and
one running request per engine makes both runs generate the same tokens, and --lr 0 keeps the weights, so every
rollout does. --save-debug-train-data keeps rollout_routed_experts as routing replay consumed it on each rank;
every sample must come back with the same tokens and a byte-identical R3 on each CP rank of its DP shard.
"""

import os
import tempfile
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.output_store_r3 import assert_same_trainer_r3, save_debug_train_data_arg
from tests.utils.dirs import get_test_data_dir, get_test_model_dir

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=700, suite="stage-c-8-gpu-h200", labels=["mooncake", "replay"], hardware=["hopper"], num_gpus=8
)

MODEL_DIR = get_test_model_dir()
DATA_DIR = get_test_data_dir()
MODEL_NAME = "Qwen3-30B-A3B-5layer"
MODEL_TYPE = "qwen3-30B-A3B-5layer"
TRAIN_GPUS = 6
ROLLOUT_GPUS = 2
CP_SIZE = 3
NUM_ROLLOUT = 3
SAMPLES_PER_ROLLOUT = 8
DUMP_DIR = Path(tempfile.mkdtemp(prefix="output_store_r3_"))


def prepare() -> None:
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download fzyzcjy/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=DATA_DIR)


def execute(mode: str) -> None:
    U = command_utils.default_config().create_backend()
    train_args = (
        f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}/ --ref-load {MODEL_DIR}/{MODEL_NAME}/ "
        "--megatron-to-hf-mode bridge "
        f"--prompt-data {DATA_DIR}/gsm8k/train.parquet --input-key messages --label-key label "
        "--apply-chat-template --rollout-shuffle --rm-type math "
        f"--num-rollout {NUM_ROLLOUT} --rollout-batch-size 2 --n-samples-per-prompt 4 "
        f"--global-batch-size {SAMPLES_PER_ROLLOUT} "
        "--rollout-max-response-len 256 --rollout-temperature 1 "
        # Same tokens in both runs: batch-invariant kernels, a seed per sample, one request at a time per engine.
        "--sglang-enable-deterministic-inference --rollout-seed 42 --sglang-max-running-requests 1 "
        "--use-rollout-routing-replay "
        f"{save_debug_train_data_arg(DUMP_DIR, mode)}"
        f"{command_utils.get_mooncake_object_store_args()} "
        f"{'--use-output-store ' if mode == 'store' else ''}"
        "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        # Unchanged weights: every rollout of both runs generates from the same model.
        "--optimizer adam --lr 0 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        # Megatron wants a multiple of 2 x CP, which Miles' 4096 placeholder is not for CP 3.
        "--seq-length 4104 "
        f"--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size {CP_SIZE} "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {TRAIN_GPUS} "
        f"--rollout-num-gpus {ROLLOUT_GPUS} --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "
        f"{command_utils.get_default_wandb_args(__file__)} "
    )
    U.execute_train(train_args=train_args, num_gpus_per_node=TRAIN_GPUS + ROLLOUT_GPUS, megatron_model_type=MODEL_TYPE)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute("inline")
    execute("store")
    assert_same_trainer_r3(DUMP_DIR, cp_size=CP_SIZE, num_samples=NUM_ROLLOUT * SAMPLES_PER_ROLLOUT)
