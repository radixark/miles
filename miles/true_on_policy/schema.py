from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrueOnPolicyContractName = Literal["qwen3_dense_hopper_bf16_true_on_policy_v1", "qwen3_dense_blackwell_bf16_true_on_policy_v1"]
ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen3_next"]
KernelContract = Literal["qwen3_dense_sglang_math"]
# Where the rollout logprobs the trainer must match come from: an SGLang prefill that re-scores each
# finished rollout, or the decode pass that sampled it (the sampler itself).
LogprobContract = Literal["sglang_prefill", "sglang_decode"]


@dataclass(frozen=True)
class TrueOnPolicyContractSchema:
    """Declarative cross-repo identity for a true-on-policy parity contract."""

    name: TrueOnPolicyContractName
    model_family: ModelFamily
    required_kernel_contracts: tuple[KernelContract, ...]
    logprob_contract: LogprobContract
    sglang_attention_backend: str
    fsdp_attention_implementation: str
    disable_megatron_sequence_parallel: bool


# Qwen3 dense on Hopper (SM90), FlashAttention-3 on both sides. The trainer matches an SGLang prefill
# re-score of each rollout; on Hopper SGLang's decode and prefill logprobs are bitwise equal, so that is
# the sampler's decode too. SGLang ships this contract as qwen3_dense_true_on_policy_v1.
QWEN3_DENSE_HOPPER_BF16_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_hopper_bf16_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_prefill",
    sglang_attention_backend="fa3",
    fsdp_attention_implementation="flash_attention_3",
    disable_megatron_sequence_parallel=True,
)


# Qwen3 dense on Blackwell (SM100), FlashAttention-4 on both sides. The trainer matches the sampler's
# decode directly (its own logprobs; nothing is re-scored); SGLang's FA4 decode and prefill logprobs are
# bitwise equal too. FSDP with single-GPU rollout engines only.
QWEN3_DENSE_BLACKWELL_BF16_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_blackwell_bf16_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_decode",
    sglang_attention_backend="fa4",
    fsdp_attention_implementation="fa4",
    disable_megatron_sequence_parallel=True,
)
