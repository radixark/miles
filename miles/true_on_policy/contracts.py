from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
    QWEN3_MOE_TRUE_ON_POLICY_V1_SCHEMA,
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
        parallel_layout=None,
        sglang_target: str | None = None,
    ) -> dict[str, object]:
        uses_megatron = train_backend == "megatron"
        if parallel_layout is not None:
            uses_tp_invariant_rollout = (
                parallel_layout.uses_train_tp
                or parallel_layout.uses_train_expert_tp
                or parallel_layout.uses_rollout_tp
            )
            uses_ep_invariant_moe = parallel_layout.uses_train_ep or parallel_layout.uses_rollout_ep
        else:
            uses_tp_invariant_rollout = sglang_target == "fsdp_tp"
            uses_ep_invariant_moe = False
        is_moe = self.model_family == "qwen3_moe"
        # SGLang attention-DP + EP is not parity-gated for Qwen3-30B-A3B yet.
        # Keep the true-on-policy launch on the verified EP-only rollout path.
        sglang_attention_data_parallel_size = 1
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
            "deterministic_moe_routing": is_moe,
            "moe_topk_tiebreak": "stable_sort" if is_moe else None,
            "deterministic_moe_dispatch": is_moe and uses_ep_invariant_moe,
            "deterministic_moe_combine": is_moe and uses_ep_invariant_moe,
            "ep_invariant_moe": is_moe and uses_ep_invariant_moe,
            "sglang_attention_data_parallel_size": sglang_attention_data_parallel_size,
            # Qwen3-MoE strict CUDA graph replay returns logits in the same dtype
            # as eager decode, so the sampler/logprob path remains exact-zero.
            "disable_sglang_cuda_graph": False,
        }


QWEN3_DENSE_TRUE_ON_POLICY_V1 = TrueOnPolicyContract(
    schema=QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
)

QWEN3_MOE_TRUE_ON_POLICY_V1 = TrueOnPolicyContract(
    schema=QWEN3_MOE_TRUE_ON_POLICY_V1_SCHEMA,
)


_CONTRACT_BY_NAME = {
    QWEN3_DENSE_TRUE_ON_POLICY_V1.name: QWEN3_DENSE_TRUE_ON_POLICY_V1,
    QWEN3_MOE_TRUE_ON_POLICY_V1.name: QWEN3_MOE_TRUE_ON_POLICY_V1,
}


def get_true_on_policy_contract(name: str) -> TrueOnPolicyContract:
    try:
        return _CONTRACT_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(f"Unsupported true-on-policy contract {name!r}. Supported contracts: {supported}") from exc
