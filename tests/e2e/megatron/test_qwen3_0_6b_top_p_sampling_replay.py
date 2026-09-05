import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=900,
    suite="stage-c-2-gpu-h200",
    labels=["megatron", "sglang", "replay"],
)
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2


def prepare() -> None:
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )


def execute() -> None:
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 128 "
        "--rollout-temperature 1 "
        "--rollout-top-p 0.8 "
        "--rollout-top-k 32 "
        "--global-batch-size 4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    megatron_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-decode-log-interval 1000 "
        "--use-miles-router "
    )

    train_args = (
        ckpt_args
        + rollout_args
        + optimizer_args
        + grpo_args
        + megatron_args
        + sglang_args
        + U.get_default_wandb_args(__file__)
        + "--ci-test --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
