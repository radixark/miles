"""Shared setup for Qwen3.5-35B-A3B e2e cases.

Cases cover speculative decoding x R3 (the {spec}_{r3} fields of each file name):
- spec_mtptrain_r3: EAGLE with MTP training on (1 layer + loss factor); the MTP/draft weights
  are synced.
- spec_nor3: EAGLE from the checkpoint draft, MTP training off. Whether the MTP layer is
  checked is a weight-check *selector* concern, not a skip-list one.
- nospec_nor3 and nospec_r3: EAGLE off, so there is no draft to sync and the selector is
  "target".

miles has no VLM/vision implementation on the training side, so Qwen3.5's `visual.*`
weights are never synced and must be excluded from the weight-equality check; each case
passes `check_weight_update_skip_list=("visual",)`.

Each case picks its own topology (see its CASE). Spec (EAGLE + spec-v2 mamba scheduler), R3,
colocation and fully-async are per-case.
"""

import os
from dataclasses import dataclass
from typing import Literal

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3.5-35B-A3B"
MODEL_TYPE = "qwen3.5-35B-A3B"


@dataclass
class CaseConfig:
    # Topology / GPU counts — explicit per case (see each file's CASE).
    num_gpus_per_node: int
    cp_size: int
    pp_size: int
    tp_size: int
    ep_size: int
    rollout_num_gpus_per_engine: int
    # Whether MTP training is enabled. On -> --enable-mtp-training --mtp-num-layers 1
    # (+ loss factor); off -> --mtp-num-layers 0, so the trainer builds no MTP layer (note
    # `--enable-mtp-training --mtp-num-layers 0` would fail the arguments.py assert).
    enable_mtp_training: bool
    # Whether to enable R3 routing replay (--use-rollout-routing-replay).
    use_r3: bool
    max_tokens_per_gpu: int = 8192
    # Weight-check selector: "all" (target + draft) or "target" (target model only; use
    # when MTP training is off so the un-synced draft is not checked).
    check_weight_update_selector: str = "all"
    # Rollout weight-name substrings to exclude from the equality check (substring match;
    # mismatches become non-fatal). Cases pass ("visual",): miles has no VLM/vision
    # implementation on the training side, so those weights are never synced.
    check_weight_update_skip_list: tuple[str, ...] = ()
    # SGLang-side DeepEP; Megatron's dispatcher is megatron_dispatcher (default flex, whose default backend is DeepEP).
    use_deepep: bool = False
    # SGLang DeepEP dispatch dtype; None keeps "auto", which dispatches FP8 even for a BF16 model
    # on the DeepGEMM runner, so a BF16-dispatch case passes "bf16".
    sglang_deepep_dispatcher_output_dtype: str = None
    # Serve Qwen/Qwen3.5-35B-A3B-FP8 against bf16 training; weight updates re-quantize.
    use_fp8_rollout: bool = False
    # EAGLE speculative decoding (MTP draft) with spec v2 in the rollout.
    use_spec: bool = True
    # None omits --sglang-ep-size, so SGLang shards the experts by TP alone (EP 1).
    sglang_ep_size: int = None
    sglang_dp_size: int = None
    sglang_enable_dp_attention: bool = False
    megatron_dispatcher: str = "flex"
    rollout_max_response_len: int = 8192
    colocate: bool = True
    rollout_num_gpus: int = None
    fully_async: bool = False
    optimizer: Literal["adam", "dist_muon"] = "adam"
    extra_args: str = ""

    def __post_init__(self):
        if self.optimizer not in ("adam", "dist_muon"):
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.fully_async and self.colocate:
            raise ValueError("fully_async requires colocate=False: train_async.py rejects colocation")
        if not self.colocate and self.rollout_num_gpus is None:
            raise ValueError("rollout_num_gpus must be set when colocate is False")
        if not self.use_spec and (self.enable_mtp_training or self.check_weight_update_selector != "target"):
            raise ValueError("without spec there is no draft: set enable_mtp_training=False and selector 'target'")
        if self.sglang_deepep_dispatcher_output_dtype is not None and not self.use_deepep:
            raise ValueError("sglang_deepep_dispatcher_output_dtype requires use_deepep=True")


def prepare(case: CaseConfig) -> None:
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    if case.use_fp8_rollout:
        U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME}-FP8 --local-dir /root/models/{MODEL_NAME}-FP8")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=case.num_gpus_per_node,
    )


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    enable_eval = os.environ.get("MILES_TEST_ENABLE_EVAL", "0").lower() in ("1", "true", "yes")

    hf_checkpoint = f"/root/models/{MODEL_NAME}-FP8" if case.use_fp8_rollout else f"/root/models/{MODEL_NAME}"
    ckpt_args = f"--hf-checkpoint {hf_checkpoint} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

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
        f"--rollout-max-response-len {case.rollout_max_response_len} "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
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
        f"--context-parallel-size {case.cp_size} "
        f"--expert-model-parallel-size {case.ep_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {case.max_tokens_per_gpu} "
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
        f"--optimizer {case.optimizer} "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if case.optimizer == "dist_muon":
        # Muon offloads through LayerWise; HybridDeviceOptimizer is Adam-only.
        optimizer_args += (
            "--chunked-optimizer-state-offload "
            "--optimizer-state-offload-fraction 1.0 "
            "--optimizer-state-offload-chunk-size-mb 1024 "
        )
    else:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    # DeepEP low-latency dispatch (every decode-phase forward, incl. EAGLE verify) holds at most
    # 128 tokens per rank. EAGLE verify feeds 3 tokens per request; DP attention splits
    # --max-running-requests over the DP ranks, and each rank dispatches its attention-TP share of
    # its DP rank's tokens, so with or without DP attention a rank dispatches at most
    # max_running * tokens_per_request / rollout_num_gpus_per_engine tokens.
    tokens_per_request = 3 if case.use_spec else 1
    max_running = (
        min(128, 128 * case.rollout_num_gpus_per_engine // tokens_per_request // 8 * 8) if case.use_deepep else 512
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine {case.rollout_num_gpus_per_engine} "
        # 0.6 (not 0.7): colocate leaves ~11GB of resident training memory on each GPU, so
        # sglang at 0.7 OOMs in the rollout MoE forward; 0.6 leaves headroom for both.
        "--sglang-mem-fraction-static 0.6 "
    )
    if case.sglang_ep_size is not None:
        sglang_args += f"--sglang-ep-size {case.sglang_ep_size} "
    sglang_args += f"--sglang-max-running-requests {max_running} "
    if case.use_spec:
        sglang_args += (
            # EAGLE speculative decoding (MTP draft)
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 2 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 3 "
            # spec v2: required to pair speculative decoding with radix cache on Qwen3.5MoE
            # (see scripts/run_qwen3_5_35b_a3b_mtp.py); also needs SGLANG_ENABLE_SPEC_V2=1.
            "--sglang-mamba-radix-cache-strategy extra_buffer "
        )
    if case.sglang_dp_size is not None:
        sglang_args += f"--sglang-data-parallel-size {case.sglang_dp_size} "
    if case.sglang_enable_dp_attention:
        sglang_args += "--sglang-enable-dp-attention "
    if case.use_r3:
        sglang_args += "--use-rollout-routing-replay "
    if case.use_deepep:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
        # The decode CUDA-graph batch is per DP rank: capture at most one DP rank's requests.
        sglang_args += f"--sglang-cuda-graph-max-bs-decode {max_running // (case.sglang_dp_size or 1)} "
        if not case.use_fp8_rollout:
            # BF16 experts have SGLang DeepEP kernels only on the DeepGEMM runner.
            sglang_args += "--sglang-moe-runner-backend deep_gemm "
        if case.sglang_deepep_dispatcher_output_dtype is not None:
            sglang_args += f"--sglang-deepep-dispatcher-output-dtype {case.sglang_deepep_dispatcher_output_dtype} "

    # When MTP training is off the rollout still runs EAGLE spec from the checkpoint
    # draft; those draft weights just never get synced (the spec_nor3 case checks only the
    # target through its selector).
    if case.enable_mtp_training:
        mtp_args = "--enable-mtp-training " "--mtp-num-layers 1 " "--mtp-loss-scaling-factor 0.2 "
    else:
        # The model script always passes --mtp-num-layers 1, and Megatron adds an MTP loss whenever the
        # layer exists, even without --enable-mtp-training; this later flag overrides it.
        mtp_args = "--mtp-num-layers 0 "

    ci_args = "--ci-test "
    ci_args += f"--check-weight-update-selector {case.check_weight_update_selector} "
    if case.check_weight_update_skip_list:
        ci_args += "--check-weight-update-skip-list " + " ".join(case.check_weight_update_skip_list) + " "
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
    if case.fully_async:
        misc_args += "--fully-async "
    misc_args += f"--moe-token-dispatcher-type {case.megatron_dispatcher} "
    if case.colocate and case.optimizer == "adam":
        misc_args += "--rematerialize-param-from-master-weight "

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
    train_args = build_train_args(case, wandb_file=wandb_file)
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus_per_node + (0 if case.colocate else case.rollout_num_gpus),
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py" if case.fully_async else "train.py",
        extra_env_vars={"SGLANG_ENABLE_SPEC_V2": "1"} if case.use_spec else {},
    )
