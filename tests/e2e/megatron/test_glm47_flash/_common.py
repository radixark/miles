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
    rollout_num_gpus_per_engine: int
    tp_size: int
    ep_size: int
    sglang_ep_size: int = None
    use_deepep: bool = False
    sglang_deepep_mode: str = "auto"
    use_fp8_rollout: bool = False
    use_int4_rollout: bool = False
    use_bridge: bool = False
    use_r3: bool = False
    max_tokens_per_gpu: int = 8192
    rollout_max_response_len: int = 8192
    colocate: bool = True
    rollout_num_gpus: int = None
    update_weight_transfer_mode: str = None
    num_rollout: int = 2
    fully_async: bool = False
    extra_args: str = ""

    def __post_init__(self):
        # Validation only — topology values are passed explicitly, not inferred.
        if self.fully_async and self.colocate:
            raise ValueError("fully_async requires colocate=False: train_async.py rejects colocation")
        if self.num_gpus_per_node % (self.cp_size * self.pp_size) != 0:
            raise ValueError(
                "num_gpus_per_node must be divisible by cp_size * pp_size: "
                f"{self.num_gpus_per_node=} {self.cp_size=} {self.pp_size=}"
            )
        if not self.colocate and self.rollout_num_gpus is None:
            raise ValueError("rollout_num_gpus must be set when colocate is False")
        rollout_pool = self.num_gpus_per_node if self.colocate else self.rollout_num_gpus
        if rollout_pool % self.rollout_num_gpus_per_engine != 0:
            raise ValueError(
                "rollout pool must be divisible by rollout_num_gpus_per_engine: "
                f"{rollout_pool=} {self.rollout_num_gpus_per_engine=}"
            )
        if self.update_weight_transfer_mode is not None:
            assert self.update_weight_transfer_mode == "broadcast"


def prepare(case: CaseConfig) -> None:
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download zai-org/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
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
    are always on for this suite; case-specific rollout and DeepEP knobs are
    exposed via CaseConfig.
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
        f"--num-rollout {case.num_rollout} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {case.rollout_max_response_len} "
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
        f"--rollout-num-gpus-per-engine {case.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.7 "
        # EAGLE speculative decoding (MTP)
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
    )

    if case.use_deepep:
        # GLM-4.7-Flash rolls out in BF16, and SGLang has DeepEP MoE kernels for BF16 experts
        # only on the DeepGEMM runner.
        sglang_args += (
            "--sglang-moe-a2a-backend deepep --sglang-moe-runner-backend deep_gemm "
            f"--sglang-deepep-mode {case.sglang_deepep_mode} "
        )
    if case.sglang_ep_size is not None:
        sglang_args += f"--sglang-expert-parallel-size {case.sglang_ep_size} "

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
    )
    if case.colocate:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {case.rollout_num_gpus} "

    if case.update_weight_transfer_mode is not None:
        misc_args += f"--update-weight-transfer-mode {case.update_weight_transfer_mode} "

    if case.fully_async:
        misc_args += "--fully-async "

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
        f"{case.extra_args} "
    )
    return train_args


def execute(case: CaseConfig, *, wandb_file: str) -> None:
    # Loosen replay mismatch threshold for GLM-4.7-Flash with MTP
    os.environ["MILES_TEST_R3_THRESHOLD"] = "0.05"

    train_args = build_train_args(case, wandb_file=wandb_file)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus_per_node + (0 if case.colocate else case.rollout_num_gpus),
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py" if case.fully_async else "train.py",
    )
