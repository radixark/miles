"""
Reproduce user-reported 5-6x R3 slowdown with:
- 1-node 8-GPU setup, EP=4 per engine (2 engines)
- Very long responses (60K tokens)
- High concurrency (1024)
- Large cuda graph batch sizes
"""

import os

import miles.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "0")))

MODEL_NAME = "Qwen3-30B-A3B-Thinking-2507"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 60000 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 5 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 60000 "
        "--eval-top-k -1 "
        "--eval-top-p 1 "
        "--eval-temperature 0 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--update-weight-buffer-size 4294967296 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 30000 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.0 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
        "--use-rollout-routing-replay "
        "--use-miles-router "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1.5e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--clip-grad 0.1 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-server-concurrency 1024 "
        "--sglang-ep-size 4 "
        "--sglang-cuda-graph-bs "
        "1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 "
        "104 112 120 128 136 144 152 160 168 176 184 192 200 "
        "208 216 224 232 240 248 256 "
        "--sglang-enable-metrics "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--distributed-timeout-minutes 30 "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        config=U.ExecuteTrainConfig(num_nodes=1),
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
