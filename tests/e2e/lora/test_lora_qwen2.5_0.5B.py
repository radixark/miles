"""E2E test for LoRA training with Qwen2.5-0.5B on GSM8K.

Uses the Megatron backend with bridge mode.  Runs a short GRPO training loop
with LoRA enabled (rank=32, all-linear) to validate:
  - LoRA model setup via Bridge
  - LoRA weight sync to SGLang rollout engines
  - LoRA checkpoint save (native + HF PEFT format)
  - Training completes without errors

Requires: 8 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import glob
import json
import os

import torch

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=400, suite="stage-c-4-gpu-h200", labels=["lora"])
register_rocm_ci(est_time=300, suite="nightly-stage-c-4-gpu-mi350", labels=["lora"])


ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4
SAVE_DIR = "/root/checkpoints/lora-qwen2.5-0.5B-ci"


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")
    U.exec_command_cpu(f"rm -rf {SAVE_DIR}")


def _assert_peft_export():
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    adapter_dirs = sorted(glob.glob(f"{SAVE_DIR}/iter_*/adapter"))
    assert adapter_dirs
    adapter_dir = adapter_dirs[-1]
    on_disk = load_file(f"{adapter_dir}/adapter_model.safetensors")
    assert on_disk
    base = AutoModelForCausalLM.from_pretrained(f"/root/models/{MODEL_NAME}", dtype=torch.float32)
    loaded = PeftModel.from_pretrained(base, adapter_dir).state_dict()

    expected_names = {name.replace(".weight", ".default.weight") for name in on_disk}
    loaded_names = {name for name in loaded if ".lora_A." in name or ".lora_B." in name}
    assert loaded_names == expected_names
    for name, expected in on_disk.items():
        loaded_name = name.replace(".weight", ".default.weight")
        assert torch.equal(loaded[loaded_name], expected), name

    with open(f"{adapter_dir}/adapter_config.json") as config_file:
        config = json.load(config_file)
    assert config["base_model_name_or_path"] == f"/root/models/{MODEL_NAME}/"


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " "--megatron-to-hf-mode bridge "

    lora_args = "--lora-rank 32 " "--lora-alpha 32 " "--lora-dropout 0.0 " '--target-modules "all-linear" '

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
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

    ci_args = "--ci-test "

    save_args = f"--save-interval 2 --save {SAVE_DIR} "

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
    _assert_peft_export()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
