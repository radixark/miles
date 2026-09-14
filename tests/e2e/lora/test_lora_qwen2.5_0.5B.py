"""E2E test for LoRA training with Qwen2.5-0.5B on GSM8K.

Uses the Megatron backend with bridge mode. Runs a short GRPO training loop
with LoRA enabled (rank=32, all-linear) to validate:
  - LoRA model setup via Bridge
  - LoRA weight sync to SGLang rollout engines
  - LoRA checkpoint save (native + HF PEFT format)
  - Rollout, LR scheduler, and dataset cursor resume
  - Training completes without errors

Requires: 4 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import os
import shutil
from pathlib import Path

import torch

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=400, suite="stage-c-4-gpu-h200", labels=["lora"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=300, suite="nightly-stage-c-4-gpu-mi350", labels=["lora"])


ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4
ROLLOUT_BATCH_SIZE = 8
N_SAMPLES_PER_PROMPT = 8
CHECKPOINT_DIR = Path("/root/checkpoints/lora-qwen2.5-0.5B-ci")


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)


def execute(adapter_path: Path | None = None):
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " "--megatron-to-hf-mode bridge "
    if adapter_path is not None:
        ckpt_args += f"--lora-adapter-path {adapter_path} --no-load-optim "

    lora_args = "--lora-rank 32 " "--lora-alpha 32 " "--lora-dropout 0.0 " '--target-modules "all-linear" '

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        f"--rollout-batch-size {ROLLOUT_BATCH_SIZE} "
        f"--n-samples-per-prompt {N_SAMPLES_PER_PROMPT} "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    eval_args = (
        f"{'--eval-interval 2 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.4 "

    ci_args = "--ci-test "
    if adapter_path is not None:
        ci_args += "--ci-disable-kl-checker --ci-disable-logprobs-checker "

    save_args = f"--save-interval 1 --save {CHECKPOINT_DIR} "

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
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{save_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def load_checkpoint_state(iteration: int):
    adapter = CHECKPOINT_DIR / f"iter_{iteration:07d}" / "adapter"
    training_state = torch.load(adapter / "training_state_rank0.pt", map_location="cpu", weights_only=True)
    data_state = torch.load(
        CHECKPOINT_DIR / "rollout" / f"global_dataset_state_dict_{iteration}.pt",
        map_location="cpu",
        weights_only=True,
    )
    return training_state, data_state


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
    resume_from = CHECKPOINT_DIR / "iter_0000001" / "adapter"
    assert resume_from.is_dir()
    training_state_before, data_state_before = load_checkpoint_state(1)
    first_checkpoint = CHECKPOINT_DIR / "iter_0000000"
    shutil.rmtree(first_checkpoint)
    shutil.rmtree(CHECKPOINT_DIR / "iter_0000002")
    (CHECKPOINT_DIR / "rollout" / "global_dataset_state_dict_2.pt").unlink()
    execute(adapter_path=resume_from)
    assert not first_checkpoint.exists()
    training_state_after, data_state_after = load_checkpoint_state(2)
    assert training_state_after["opt_param_scheduler"]["num_steps"] == (
        training_state_before["opt_param_scheduler"]["num_steps"] + ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT
    )
    assert data_state_after["sample_offset"] == data_state_before["sample_offset"] + ROLLOUT_BATCH_SIZE
