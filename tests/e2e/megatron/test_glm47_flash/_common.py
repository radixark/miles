import os
from dataclasses import dataclass

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "GLM-4.7-Flash"
MODEL_TYPE = "glm4.7-flash"

TIGHT_HOST_MEMORY = bool(int(os.environ.get("MILES_TEST_TIGHT_HOST_MEMORY", "1")))


@dataclass
class CaseConfig:
    num_gpus_per_node: int
    cp_size: int
    pp_size: int
    tp_size: int = None
    ep_size: int = None
    use_deepep: bool = False
    use_fp8_rollout: bool = False
    use_int4_rollout: bool = False
    use_bridge: bool = False
    use_r3: bool = False
    max_tokens_per_gpu: int = 8192

    def __post_init__(self):
        if self.tp_size is None:
            self.tp_size = self.num_gpus_per_node // self.cp_size // self.pp_size
        if self.ep_size is None:
            self.ep_size = self.num_gpus_per_node // self.pp_size


def prepare(case: CaseConfig) -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download zai-org/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=case.num_gpus_per_node,
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
        "--num-rollout 2 "
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

    perf_args = (
        f"--tensor-model-parallel-size {case.tp_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {case.pp_size} "
        f"{'--decoder-last-pipeline-num-layers 23 ' if case.pp_size == 2 else ''}"
        f"--context-parallel-size {case.cp_size} "
        f"--expert-model-parallel-size {case.ep_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {case.max_tokens_per_gpu} "
    )

    if TIGHT_HOST_MEMORY:
        perf_args += "--exp-avg-dtype fp16 "
        perf_args += "--exp-avg-sq-dtype fp16 "
        perf_args += "--main-params-dtype fp16 "

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
        "--sglang-mem-fraction-static 0.7 "
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
        f"--actor-num-gpus-per-node {case.num_gpus_per_node} "
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
    # Loosen replay mismatch threshold for GLM-4.7-Flash with MTP
    os.environ["MILES_TEST_R3_THRESHOLD"] = "0.05"

    train_args = build_train_args(case, wandb_file=wandb_file)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus_per_node,
        megatron_model_type=MODEL_TYPE,
    )
