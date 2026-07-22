import os

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# MoE-expert LoRA smoke test on gpt-oss-20b (expert-only targets, bridge mode; CI-sized
# version of examples/lora/run-gpt-oss-20B-megatron-moe-lora.sh). Runs both serving
# combinations — {shared-outer + virtual-experts, per-expert + no-virtual-experts} — and
# every combination must pass. Functionality, not accuracy; 4 GPUs (TP=4, EP=1).


register_cuda_ci(est_time=1600, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts", "lora"])

MODEL_NAME = "gpt-oss-20b-bf16"
MODEL_TYPE = "gpt-oss-20b"
NUM_GPUS = 4

# (name, experts_shared_outer_loras, virtual_experts_serving)
_CONFIGS = [
    ("shared-outer + virtual-experts", True, True),
    ("per-expert + no-virtual-experts", False, False),
]


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download lmsys/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k")


def execute(shared_outer: bool, virtual_experts: bool):
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " "--megatron-to-hf-mode bridge "

    lora_args = (
        "--lora-rank 32 "
        "--lora-alpha 32 "
        "--lora-dropout 0.0 "
        '--target-modules "gate_proj,up_proj,down_proj" '
        "--sglang-lora-backend triton "
        f"{'--experts-shared-outer-loras ' if shared_outer else ''}"
        f"{'' if virtual_experts else '--no-sglang-lora-use-virtual-experts '}"
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 1 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {NUM_GPUS} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 4096 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {NUM_GPUS} "
        "--sglang-dtype bfloat16 "
        "--sglang-mem-fraction-static 0.2 "
        "--sglang-moe-runner-backend triton "
        "--sglang-decode-log-interval 1000 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--qkv-format bshd "
        "--attention-backend fused "
        "--update-weight-buffer-size 536870912 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--ci-test "
        "--ci-disable-logprobs-checker "
        "--disable-weights-backuper "
    )

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {perf_args} {optimizer_args} {grpo_args} {sglang_args} {misc_args}"
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    for name, shared_outer, virtual_experts in _CONFIGS:
        print(f"[gpt-oss-moe-lora-ci] ===== combo: {name} =====", flush=True)
        # fresh ray/sglang between combos
        U.exec_command("ray stop --force || true; pkill -9 sglang || true; sleep 10")
        execute(shared_outer, virtual_experts)
        print(f"[gpt-oss-moe-lora-ci] ===== combo PASSED: {name} =====", flush=True)
