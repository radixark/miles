import os
from dataclasses import dataclass

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "GLM-4.7-Flash"
MODEL_TYPE = "glm4.7-flash"
NUM_GPUS = 4


@dataclass(frozen=True)
class CaseConfig:
    use_deepep: bool


def prepare() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download zai-org/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
    )


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    """Build the train_args string for `case`.

    MTP (EAGLE speculative decoding) and R3 (`--use-rollout-routing-replay`)
    are always on for this suite; the only knob exposed via CaseConfig is
    whether DeepEP is used for MoE token dispatch.
    """
    enable_eval = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "0")))

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if enable_eval else ''}"
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    # tp=4 because GLM-4.7-Flash has 20 attention heads (tp must divide num_heads)
    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-rollout-routing-replay "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.8 "
        # EAGLE speculative decoding (MTP)
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
    )

    if case.use_deepep:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "

    mtp_args = "--enable-mtp-training " "--mtp-loss-scaling-factor 0.2 "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--colocate "
    )

    if case.use_deepep:
        misc_args += "--moe-token-dispatcher-type flex --moe-enable-deepep "
    else:
        misc_args += "--moe-token-dispatcher-type alltoall "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(wandb_file)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{mtp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )
    return train_args


def execute(case: CaseConfig, *, wandb_file: str) -> None:
    # Set replay_check_threshold to 1e-1 for GLM-4.7-Flash with MTP
    os.environ["MILES_TEST_R3_THRESHOLD"] = "0.05"

    train_args = build_train_args(case, wandb_file=wandb_file)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )
