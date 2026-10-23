"""The R3 the trainer consumes is the same through the output store as inline, for a deterministic rollout.

Two one-rollout runs differ only in --use-output-store: a 5-layer Qwen3-30B-A3B slice, one Megatron trainer GPU
and one SGLang engine GPU, deterministic inference with fixed per-sample seeds and one running request at a time,
so both runs generate the same tokens. --save-debug-train-data keeps rollout_routed_experts as routing replay
consumed it; every sample must come back with the same tokens and a byte-identical R3.
"""

import os
import tempfile
from pathlib import Path

import torch
from tests.ci.ci_register import register_cuda_ci
from tests.utils.dirs import get_test_data_dir, get_test_model_dir

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=800, suite="stage-c-2-gpu-h200", labels=["mooncake"], hardware=["hopper"], num_gpus=2)

MODEL_DIR = get_test_model_dir()
DATA_DIR = get_test_data_dir()
MODEL_NAME = "Qwen3-30B-A3B-5layer"
MODEL_TYPE = "qwen3-30B-A3B-5layer"
NUM_GPUS = 2
NUM_SAMPLES = 8
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
        f"--num-rollout 1 --rollout-batch-size 2 --n-samples-per-prompt 4 --global-batch-size {NUM_SAMPLES} "
        "--rollout-max-response-len 256 --rollout-temperature 1 "
        # Same tokens in both runs: batch-invariant kernels, a seed per sample, one request at a time.
        "--sglang-enable-deterministic-inference --rollout-seed 42 --sglang-max-running-requests 1 "
        "--use-rollout-routing-replay "
        f"--save-debug-train-data {DUMP_DIR}/{mode}/{{rollout_id}}_{{rank}}.pt "
        f"{command_utils.get_mooncake_object_store_args()} "
        f"{'--use-output-store ' if mode == 'store' else ''}"
        "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "
        f"{command_utils.get_default_wandb_args(__file__)} "
    )
    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)


def trainer_r3(mode: str) -> dict[tuple[int, ...], torch.Tensor]:
    # One trainer rank and one rollout: one dump.
    [path] = (DUMP_DIR / mode).glob("*.pt")
    rollout_data = torch.load(path, map_location="cpu", weights_only=False)["rollout_data"]
    return {
        tuple(tokens.tolist()): r3
        for tokens, r3 in zip(rollout_data["tokens"], rollout_data["rollout_routed_experts"], strict=True)
    }


def compare() -> None:
    inline, store = trainer_r3("inline"), trainer_r3("store")
    assert len(inline) == NUM_SAMPLES, f"expected {NUM_SAMPLES} distinct samples, got {len(inline)}"
    assert inline.keys() == store.keys(), "the two rollouts generated different tokens; R3 cannot be compared"
    for tokens, expected in inline.items():
        actual = store[tokens]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, (actual.dtype, actual.shape)
        assert torch.equal(actual, expected), f"R3 differs for the {len(tokens)}-token sample"


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute("inline")
    execute("store")
    compare()
