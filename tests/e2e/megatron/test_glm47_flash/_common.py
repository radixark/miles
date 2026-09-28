import os
from dataclasses import dataclass

from miles.utils.external_utils import command_utils

MODEL_NAME = "GLM-4.7-Flash"
MODEL_TYPE = "glm4.7-flash"

TIGHT_HOST_MEMORY = bool(int(os.environ.get("MILES_TEST_TIGHT_HOST_MEMORY", "1")))

GLOBAL_BATCH_SIZE = 32

# SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK default: the most tokens one rank dispatches per
# step in DeepEP low-latency mode.
DEEPEP_LOW_LATENCY_MAX_TOKENS_PER_RANK = 128


@dataclass
class CaseConfig:
    num_gpus_per_node: int
    cp_size: int
    pp_size: int
    rollout_num_gpus_per_engine: int
    tp_size: int
    ep_size: int
    sglang_ep_size: int = None
    sglang_dp_size: int = None
    sglang_enable_dp_attention: bool = False
    use_deepep: bool = False
    sglang_deepep_mode: str = "auto"
    # None keeps "auto", which dispatches FP8 even for a BF16 model on the DeepGEMM runner.
    sglang_deepep_dispatcher_output_dtype: str = None
    # Serve RadixArk/glm47-flash-blockwise-fp8 against BF16 training; weight updates re-quantize.
    use_fp8_rollout: bool = False
    use_int4_rollout: bool = False
    use_bridge: bool = False
    # EAGLE from the checkpoint's MTP layer, with MTP training on so the draft weights are synced.
    # Off: no draft runs, the trainer builds no MTP layer, and the weight check covers the target only.
    use_spec: bool = True
    use_r3: bool = True
    # "bshd" (Megatron only) trains fixed micro-batches of micro_batch_size samples instead of
    # dynamic THD batches.
    qkv_format: str = "thd"
    micro_batch_size: int = None
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
        if self.sglang_enable_dp_attention and (
            self.sglang_dp_size is None or self.rollout_num_gpus_per_engine % self.sglang_dp_size != 0
        ):
            raise ValueError(
                "SGLang DP attention needs sglang_dp_size dividing the engine TP: "
                f"{self.rollout_num_gpus_per_engine=} {self.sglang_dp_size=}"
            )
        # At attention TP 4 the checkpoint's kv_b_proj shard is 5 x (192 + 256) = 2240 rows, not a
        # multiple of its 128-row FP8 blocks, so SGLang cannot load it.
        if self.use_fp8_rollout and self.sglang_attn_tp_size > 2:
            raise ValueError(
                f"the blockwise FP8 checkpoint needs SGLang attention TP <= 2: {self.sglang_attn_tp_size=}"
            )
        if self.sglang_deepep_dispatcher_output_dtype is not None and not self.use_deepep:
            raise ValueError("sglang_deepep_dispatcher_output_dtype requires use_deepep=True")
        if (self.qkv_format == "bshd") != (self.micro_batch_size is not None):
            raise ValueError("micro_batch_size is the BSHD batch; THD batches dynamically by max_tokens_per_gpu")
        if self.qkv_format == "bshd":
            # micro_batch_size > 1 stacks several samples in the BSHD batch dimension; Miles pads every
            # sample to the longest on its data-parallel rank either way.
            dp_size = self.num_gpus_per_node // (self.tp_size * self.cp_size * self.pp_size)
            if self.micro_batch_size < 2 or GLOBAL_BATCH_SIZE // dp_size < self.micro_batch_size:
                raise ValueError(
                    "BSHD needs micro_batch_size > 1 and at least that many samples per data-parallel rank "
                    f"per step: {self.micro_batch_size=} {GLOBAL_BATCH_SIZE=} {dp_size=}"
                )

    @property
    def sglang_attn_tp_size(self) -> int:
        return self.rollout_num_gpus_per_engine // (self.sglang_dp_size if self.sglang_enable_dp_attention else 1)


def prepare(case: CaseConfig) -> None:
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download zai-org/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    if case.use_fp8_rollout:
        U.exec_command_cpu(f"hf download RadixArk/glm47-flash-blockwise-fp8 --local-dir /root/models/{MODEL_NAME}-FP8")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=case.num_gpus_per_node,
    )


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    """Build the train_args string for `case`.

    Speculative decoding (EAGLE + MTP training), R3 (`--use-rollout-routing-replay`), FP8
    rollout, SGLang DP attention, DeepEP and the BSHD layout are per-case knobs in CaseConfig.
    """
    enable_eval = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "0")))

    # The BF16 model still backs the torch_dist conversion and --ref-load.
    hf_checkpoint = f"/root/models/{MODEL_NAME}-FP8" if case.use_fp8_rollout else f"/root/models/{MODEL_NAME}"
    ckpt_args = f"--hf-checkpoint {hf_checkpoint} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

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
        f"--global-batch-size {GLOBAL_BATCH_SIZE} "
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
    )
    if case.qkv_format == "bshd":
        # BSHD rejects dynamic batching, and --max-tokens-per-gpu only sizes dynamic batches.
        perf_args += f"--qkv-format bshd --micro-batch-size {case.micro_batch_size} "
    else:
        perf_args += f"--use-dynamic-batch-size --max-tokens-per-gpu {case.max_tokens_per_gpu} "

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
    )
    if case.use_r3:
        grpo_args += "--use-rollout-routing-replay "

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
        f"--rollout-num-gpus-per-engine {case.rollout_num_gpus_per_engine} " "--sglang-mem-fraction-static 0.7 "
    )
    if case.use_spec:
        sglang_args += (
            # EAGLE speculative decoding (MTP)
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 2 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 3 "
        )
    if case.sglang_dp_size is not None:
        sglang_args += f"--sglang-dp-size {case.sglang_dp_size} "
    if case.sglang_enable_dp_attention:
        sglang_args += "--sglang-enable-dp-attention "

    if case.use_deepep:
        sglang_args += f"--sglang-moe-a2a-backend deepep --sglang-deepep-mode {case.sglang_deepep_mode} "
        if not case.use_fp8_rollout:
            # SGLang has DeepEP MoE kernels for BF16 experts only on the DeepGEMM runner; blockwise
            # FP8 experts resolve the auto runner to DeepGEMM themselves.
            sglang_args += "--sglang-moe-runner-backend deep_gemm "
        if case.sglang_deepep_dispatcher_output_dtype is not None:
            sglang_args += f"--sglang-deepep-dispatcher-output-dtype {case.sglang_deepep_dispatcher_output_dtype} "
        if case.sglang_deepep_mode != "normal":
            # Every decode step runs DeepEP low-latency, where a rank dispatches its DP rank's batch
            # times the tokens each request verifies. The decode CUDA-graph batch is per DP rank (512
            # by default on H200), and --max-running-requests is split over the DP ranks.
            tokens_per_request = 3 if case.use_spec else 1
            decode_bs = DEEPEP_LOW_LATENCY_MAX_TOKENS_PER_RANK // tokens_per_request
            attn_dp_size = case.sglang_dp_size if case.sglang_enable_dp_attention else 1
            sglang_args += (
                f"--sglang-max-running-requests {decode_bs * attn_dp_size} "
                f"--sglang-cuda-graph-max-bs-decode {decode_bs} "
            )
    if case.sglang_ep_size is not None:
        sglang_args += f"--sglang-expert-parallel-size {case.sglang_ep_size} "

    if case.use_spec:
        mtp_args = "--enable-mtp-training " "--mtp-loss-scaling-factor 0.2 "
    else:
        # The model script always passes --mtp-num-layers 1, and Megatron adds an MTP loss whenever the
        # layer exists, even without --enable-mtp-training; this later flag overrides it.
        mtp_args = "--mtp-num-layers 0 "

    ci_args = "--ci-test "
    if not case.use_spec:
        ci_args += "--check-weight-update-selector target "
    if case.use_fp8_rollout:
        ci_args += "--check-weight-update-allow-quant-error "

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
        f"{command_utils.get_default_wandb_args(wandb_file)} "
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
    U = command_utils.default_config().create_backend()
    if case.use_r3:
        # Loosen replay mismatch threshold for GLM-4.7-Flash
        os.environ["MILES_TEST_R3_THRESHOLD"] = "0.05"

    train_args = build_train_args(case, wandb_file=wandb_file)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus_per_node + (0 if case.colocate else case.rollout_num_gpus),
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py" if case.fully_async else "train.py",
    )
