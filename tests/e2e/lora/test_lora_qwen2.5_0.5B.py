"""E2E test for LoRA training with Qwen2.5-0.5B on GSM8K.

Uses the Megatron backend with bridge mode.  Runs a short GRPO training loop
with LoRA enabled (rank=32, all-linear) to validate:
  - LoRA model setup via Bridge
  - LoRA weight sync to SGLang rollout engines
  - a run with --save completes (the saved iter_N/adapter is not checked here)
  - --save-hf writes a PEFT adapter through the adapter slots' probe
  - Training completes without errors

Requires: 2 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import glob
import os
import shutil

from safetensors import safe_open
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=400, suite="stage-c-2-gpu-h200", labels=["lora"], hardware=["hopper", "blackwell"], num_gpus=2
)
register_rocm_ci(est_time=500, suite="nightly-stage-c-2-gpu-mi350", labels=["lora"])


ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2
SAVE_HF_DIR = "/root/checkpoints/lora-qwen2.5-0.5B-ci-hf"


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")


def execute():
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " "--megatron-to-hf-mode bridge "

    lora_args = "--lora-rank 32 " "--lora-alpha 32 " "--lora-dropout 0.0 " '--target-modules "all-linear" '

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
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

    ci_args = "--ci-test --no-enable-sample-ownership-checker "

    save_args = (
        "--save-interval 2 "
        "--save /root/checkpoints/lora-qwen2.5-0.5B-ci "
        f"--save-hf {SAVE_HF_DIR}/rollout_{{rollout_id}} "
    )

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
        f"{command_utils.get_default_wandb_args(__file__)} "
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


def verify_hf_adapter():
    # a periodic --save-hf only logs its failure and removes the dir, so the run passing proves nothing
    adapter_dirs = glob.glob(f"{SAVE_HF_DIR}/*/adapter")
    assert adapter_dirs, f"--save-hf wrote no adapter under {SAVE_HF_DIR}"
    for adapter_dir in adapter_dirs:
        assert os.path.isfile(f"{adapter_dir}/adapter_config.json"), adapter_dir
        with safe_open(f"{adapter_dir}/adapter_model.safetensors", framework="pt") as f:
            keys = list(f.keys())
        assert keys and all(key.startswith("base_model.model.") for key in keys), keys[:4]


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    shutil.rmtree(SAVE_HF_DIR, ignore_errors=True)
    execute()
    verify_hf_adapter()
