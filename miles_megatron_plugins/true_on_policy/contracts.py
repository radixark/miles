from __future__ import annotations

from dataclasses import dataclass

from .schema import QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA, TrueOnPolicyContractName, TrueOnPolicyContractSchema

QWEN3_DENSE_TRUE_ON_POLICY_V1 = QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA.name


def _cp_comm_type_uses_a2a(cp_comm_type) -> bool:
    if isinstance(cp_comm_type, str):
        return cp_comm_type == "a2a"
    if isinstance(cp_comm_type, list):
        return any(item == "a2a" for item in cp_comm_type)
    return False


@dataclass(frozen=True)
class MegatronTrueOnPolicyRuntimePolicy:
    """Megatron-local behavior implied by a true-on-policy parity contract."""

    contract_name: str | None
    enabled: bool
    use_sglang_backend: bool
    batch_invariant_mode: bool
    disable_rope_fusion: bool
    disable_bias_swiglu_fusion: bool
    attention_backend: str
    cp_layout: str | None
    cast_attention_input_to_dense_math_dtype: bool
    cast_qk_norm_input_before_weight_mul: bool
    cast_qk_after_rope_to_dense_math_dtype: bool
    cast_lm_head_input_to_weight_dtype: bool
    deterministic_row_parallel_reduce: bool
    defer_ulysses_cp_loss_scaling_to_grad_sum: bool
    apply_logits_contract: bool
    use_sglang_final_norm: bool
    use_sglang_residual_pair: bool
    use_ulysses_cp_recompute_fallback: bool


DEFAULT_RUNTIME_POLICY = MegatronTrueOnPolicyRuntimePolicy(
    contract_name=None,
    enabled=False,
    use_sglang_backend=False,
    batch_invariant_mode=False,
    disable_rope_fusion=False,
    disable_bias_swiglu_fusion=False,
    attention_backend="default",
    cp_layout=None,
    cast_attention_input_to_dense_math_dtype=False,
    cast_qk_norm_input_before_weight_mul=True,
    cast_qk_after_rope_to_dense_math_dtype=False,
    cast_lm_head_input_to_weight_dtype=False,
    deterministic_row_parallel_reduce=False,
    defer_ulysses_cp_loss_scaling_to_grad_sum=False,
    apply_logits_contract=False,
    use_sglang_final_norm=False,
    use_sglang_residual_pair=False,
    use_ulysses_cp_recompute_fallback=False,
)


@dataclass(frozen=True)
class MegatronTrueOnPolicyContract:
    """Megatron-local adapter from a shared contract schema to runtime policy."""

    schema: TrueOnPolicyContractSchema

    @property
    def name(self) -> TrueOnPolicyContractName:
        return self.schema.name

    def policy_for(self, config) -> MegatronTrueOnPolicyRuntimePolicy:
        uses_ulysses_cp = getattr(config, "context_parallel_size", 1) > 1 and _cp_comm_type_uses_a2a(
            getattr(config, "cp_comm_type", None)
        )
        return MegatronTrueOnPolicyRuntimePolicy(
            contract_name=self.name,
            enabled=True,
            use_sglang_backend=True,
            batch_invariant_mode=getattr(config, "batch_invariant_mode", False),
            disable_rope_fusion=True,
            disable_bias_swiglu_fusion=True,
            attention_backend="fa3_varlen",
            cp_layout="ulysses_a2a" if uses_ulysses_cp else None,
            cast_attention_input_to_dense_math_dtype=True,
            cast_qk_norm_input_before_weight_mul=True,
            cast_qk_after_rope_to_dense_math_dtype=True,
            cast_lm_head_input_to_weight_dtype=True,
            deterministic_row_parallel_reduce=True,
            defer_ulysses_cp_loss_scaling_to_grad_sum=True,
            apply_logits_contract=True,
            use_sglang_final_norm=True,
            use_sglang_residual_pair=True,
            use_ulysses_cp_recompute_fallback=uses_ulysses_cp,
        )


QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT = MegatronTrueOnPolicyContract(schema=QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA)


_CONTRACT_BY_NAME = {QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT.name: QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT}


def get_true_on_policy_contract(contract_name: str) -> MegatronTrueOnPolicyContract:
    try:
        return _CONTRACT_BY_NAME[contract_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(
            f"Unsupported Megatron true-on-policy contract {contract_name!r}. " f"Supported contracts: {supported}"
        ) from exc


def validate_true_on_policy_contract(contract_name: str | None) -> None:
    if contract_name is None:
        return
    get_true_on_policy_contract(contract_name)


def resolve_true_on_policy_runtime_policy(config) -> MegatronTrueOnPolicyRuntimePolicy:
    contract_name = getattr(config, "true_on_policy_contract", None)
    if contract_name is None:
        return DEFAULT_RUNTIME_POLICY

    validate_true_on_policy_contract(contract_name)
    return get_true_on_policy_contract(contract_name).policy_for(config)
