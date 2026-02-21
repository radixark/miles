import os
from pathlib import Path

import torch

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4
DUMP_DIR = "/tmp/test_miles_dumper"

PHASES = {
    "inference": "engine_*",
    "fwd_only": "fwd_only",
    "fwd_bwd": "fwd_bwd",
}


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.exec_command(f"rm -rf {DUMP_DIR}")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type math "
        "--num-rollout 1 --rollout-batch-size 4 --n-samples-per-prompt 2 "
        "--rollout-max-response-len 128 --rollout-temperature 0.8 "
        "--global-batch-size 8 "
    )

    optimizer_args = "--optimizer adam --lr 1e-6 --lr-decay-style constant "
    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "

    perf_args = (
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6 "

    dumper_args = f"--dumper-enable --dumper-dir {DUMP_DIR} "

    misc_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate " "--megatron-to-hf-mode bridge "

    train_args = " ".join(
        [
            ckpt_args,
            rollout_args,
            optimizer_args,
            grpo_args,
            perf_args,
            sglang_args,
            dumper_args,
            misc_args,
            U.get_default_wandb_args(__file__),
        ]
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def _check_dump_dir(phase_dir: Path, exp_pattern: str) -> None:
    assert phase_dir.exists(), f"Missing dump dir: {phase_dir}"
    dump_subdirs = list(phase_dir.glob(exp_pattern))
    assert len(dump_subdirs) > 0, f"No {exp_pattern} subdirs in {phase_dir}"
    dump_files = list(dump_subdirs[0].glob("*.pt"))
    assert len(dump_files) > 0, f"No .pt files in {dump_subdirs[0]}"
    sample = torch.load(dump_files[0], weights_only=False)
    assert isinstance(sample, dict), f"Unexpected type: {type(sample)}"
    assert "value" in sample and "meta" in sample, f"Missing keys: {sample.keys()}"


def verify():
    base = Path(DUMP_DIR)

    for phase, exp_pattern in PHASES.items():
        _check_dump_dir(base / phase, exp_pattern)

    print("All dump verifications passed!")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
    verify()
