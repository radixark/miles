from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrueOnPolicyContractName = Literal["qwen3_dense_true_on_policy_v1"]
ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen3_next"]
ParallelLayout = Literal["tp", "pp", "dp", "ulysses_cp"]
LogprobContract = Literal["sglang_prefill"]
KernelContract = Literal["qwen3_dense_sglang_math"]


@dataclass(frozen=True)
class TrueOnPolicyContract:
    """Internal parity contract selected by Miles and implemented by each backend."""

    name: TrueOnPolicyContractName
    model_family: ModelFamily
    required_kernel_contracts: tuple[KernelContract, ...]
    logprob_contract: LogprobContract
    sglang_attention_backend: str
    fsdp_attention_implementation: str
    disable_megatron_sequence_parallel: bool


QWEN3_DENSE_TRUE_ON_POLICY_V1 = TrueOnPolicyContract(
    name="qwen3_dense_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_prefill",
    sglang_attention_backend="fa3",
    fsdp_attention_implementation="flash_attention_3",
    disable_megatron_sequence_parallel=True,
)


_CONTRACT_BY_NAME = {
    QWEN3_DENSE_TRUE_ON_POLICY_V1.name: QWEN3_DENSE_TRUE_ON_POLICY_V1,
}


def get_true_on_policy_contract(name: str) -> TrueOnPolicyContract:
    try:
        return _CONTRACT_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(
            f"Unsupported true-on-policy contract {name!r}. Supported contracts: {supported}"
        ) from exc
