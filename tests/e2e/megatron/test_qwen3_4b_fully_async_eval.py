"""E2E: every fully-async eval posture on Qwen3-4B, one short GRPO run each.

Three cases, in the order of how much machinery each adds:

1. ``shared``   — no eval GPUs: eval runs on the rollout engines and the producer
                  pauses for its duration (``FullyAsyncRolloutFn._call_eval``).
2. ``fleet``    — ``--eval-num-gpus 1``: a dedicated in-job engine evaluates HF
                  snapshots staged by the EvalDispatcher; training never blocks.
3. ``external`` — ``ExternalSglangEvalFn``: the same checkpoint contract served by
                  a self-launched sglang server outside the Ray placement group.

Each case is a full ``train_async.py --fully-async`` run, so a crash anywhere in
the dispatch/export/drain path fails the case; the gated train metrics come from
the same runs.

Requires: 8 GPUs, Qwen3-4B, GSM8K. Triggered by label: run-ci-megatron or run-ci-eval.
"""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["megatron", "eval", "fully-async"])
register_rocm_ci(est_time=1500, suite="nightly-stage-c-8-gpu-mi350", labels=["megatron", "eval", "fully-async"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

FEW_GPU = U.get_bool_env_var("MILES_TEST_FEW_GPU", "0")

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 4 if FEW_GPU else 8
ACTOR_GPUS = 2

EVAL_MODES = ("shared", "fleet", "external")


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
        hf_checkpoint=f"/root/models/{MODEL_NAME}",
        megatron_path="/root/Megatron-LM",
    )


def execute(eval_mode: str):
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--fully-async "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--balance-data "
        "--pause-generation-mode in_place "
    )

    eval_args = (
        "--eval-interval 1 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )
    # The rollout pool shrinks by one engine whenever the eval posture claims a GPU.
    rollout_num_gpus = NUM_GPUS - ACTOR_GPUS
    eval_env = {}
    if eval_mode != "shared":
        eval_args += "--eval-hf-dir /dev/shm/miles_eval_hf --eval-keep-snapshots 2 "
        rollout_num_gpus -= 1
    if eval_mode == "fleet":
        eval_args += "--eval-num-gpus 1 --eval-num-gpus-per-engine 1 "
    elif eval_mode == "external":
        eval_args += "--eval-function-path examples.infra_features.fully_async.external_eval_fn.ExternalSglangEvalFn "
        eval_env = {"MILES_EXTERNAL_EVAL_GPUS": str(NUM_GPUS - 1)}

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
        "--skip-actor-forward-only "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "

    ci_args = (
        "--ci-test --ci-metric-checker-key eval/gsm8k --ci-metric-checker-threshold 0.4 "
        "--ci-metric-checker-policy all "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {ACTOR_GPUS} "
        f"--rollout-num-gpus {rollout_num_gpus} "
    )

    train_args = (
        f"{ckpt_args} {rollout_args} {eval_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__)} {perf_args} {sglang_args} {ci_args} {misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars={**eval_env},
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare()
    for mode in EVAL_MODES:
        print(f"===== fully-async eval mode: {mode} =====", flush=True)
        execute(mode)
