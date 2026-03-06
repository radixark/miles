"""Launch a standard FT training run for E2E fault-tolerance tests.

Structurally mirrors tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py
but uses ExecuteTrainConfig(full_fault_tolerance=True) so the launcher
creates an FtController alongside the training job.

Requires MILES_SCRIPT_EXTERNAL_RAY=1 (uses an existing Ray cluster).
"""

import os
from dataclasses import dataclass

import miles.utils.external_utils.command_utils as U
from miles.utils.external_utils.command_utils import ExecuteTrainConfig

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"


@dataclass
class ScriptArgs(ExecuteTrainConfig):
    num_nodes: int = 2  # 2 training nodes + 1 spare for eviction in 3-node cluster
    tight_device_memory: bool = U.get_bool_env_var("MILES_TEST_TIGHT_DEVICE_MEMORY", "1")
    few_gpu: bool = U.get_bool_env_var("MILES_TEST_FEW_GPU", "1")
    full_fault_tolerance: bool = True

    @property
    def num_gpus(self) -> int:
        return 4 if self.few_gpu else 8


def prepare() -> None:
    U.exec_command_all_ray_node("mkdir -p /root/models /root/datasets")
    U.exec_command_all_ray_node(
        f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}"
    )
    U.exec_command_all_ray_node(
        "hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k"
    )


def execute(args: ScriptArgs) -> None:
    num_gpus = args.num_gpus

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-interval 20 "
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
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.6 if args.tight_device_memory else 0.7} "
        "--sglang-enable-metrics "
    )

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {num_gpus} "
        "--colocate "
        "--megatron-to-hf-mode bridge "
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
        f"{fault_tolerance_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=MODEL_TYPE,
        config=args,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


if __name__ == "__main__":
    assert U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), "MILES_SCRIPT_EXTERNAL_RAY must be set"
    args = ScriptArgs()
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
