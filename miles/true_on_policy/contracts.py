from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
    KernelContract,
    LogprobContract,
    ModelFamily,
    TrueOnPolicyContractName,
    TrueOnPolicyContractSchema,
)


@dataclass(frozen=True)
class TrueOnPolicyContract:
    """Internal parity contract selected by Miles and implemented by each backend."""

    schema: TrueOnPolicyContractSchema

    @property
    def name(self) -> TrueOnPolicyContractName:
        return self.schema.name

    @property
    def model_family(self) -> ModelFamily:
        return self.schema.model_family

    @property
    def required_kernel_contracts(self) -> tuple[KernelContract, ...]:
        return self.schema.required_kernel_contracts

    @property
    def logprob_contract(self) -> LogprobContract:
        return self.schema.logprob_contract

    @property
    def sglang_attention_backend(self) -> str:
        return self.schema.sglang_attention_backend

    @property
    def fsdp_attention_implementation(self) -> str:
        return self.schema.fsdp_attention_implementation

    @property
    def disable_megatron_sequence_parallel(self) -> bool:
        return self.schema.disable_megatron_sequence_parallel

    def kernel_policy_kwargs_for(
        self,
        *,
        train_backend: str,
        sglang_target: str,
    ) -> dict[str, object]:
        uses_megatron = train_backend == "megatron"
        uses_tp_invariant_rollout = sglang_target == "fsdp_tp"
        return {
            "deterministic_inference": True,
            "deterministic_training": True,
            "sglang_attention_backend": self.sglang_attention_backend,
            "megatron_uses_sglang_backend": uses_megatron,
            "disable_rope_fusion": uses_megatron,
            "disable_bias_swiglu_fusion": uses_megatron,
            "batch_invariant_mode": uses_megatron,
            "tp_invariant_row_linear": uses_tp_invariant_rollout,
            "deterministic_tp_allreduce": uses_tp_invariant_rollout,
        }


QWEN3_DENSE_TRUE_ON_POLICY_V1 = TrueOnPolicyContract(
    schema=QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
)


_CONTRACT_BY_NAME = {
    QWEN3_DENSE_TRUE_ON_POLICY_V1.name: QWEN3_DENSE_TRUE_ON_POLICY_V1,
}


def get_true_on_policy_contract(name: str) -> TrueOnPolicyContract:
    try:
        return _CONTRACT_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(f"Unsupported true-on-policy contract {name!r}. Supported contracts: {supported}") from exc
