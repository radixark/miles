"""Small configuration fixture for provider and checkpoint functionality checks."""

import torch
from megatron.core.transformer.enums import AttnBackend

from miles_plugins.models.muse_spark.config import map_to_megatron_fields
from miles_plugins.models.muse_spark.provider import validate_model_config
from miles_plugins.models.muse_spark.transformer_config import MuseSpark1xTransformerConfig

_RUNTIME_FIELDS = frozenset(
    {
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
        "expert_model_parallel_size",
        "context_parallel_size",
        "cp_comm_type",
        "sequence_parallel",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
        "deterministic_mode",
        "calculate_per_token_loss",
        "variable_seq_lengths",
        "use_cpu_initialization",
        "perform_initialization",
        "attention_backend",
    }
)


def build_model_config(spark, *, params_dtype=None, **runtime):
    """Separate immutable HF semantics from supported training runtime options."""
    unknown = set(runtime) - _RUNTIME_FIELDS
    if unknown:
        raise ValueError(f"unsupported Spark runtime fields: {sorted(unknown)}")
    if isinstance(runtime.get("cp_comm_type"), (list, tuple)):
        # Megatron's --cp-comm-type is a list: one entry applies to every layer, otherwise one per layer.
        comm = list(runtime["cp_comm_type"])
        runtime["cp_comm_type"] = comm[0] if len(comm) == 1 else comm
    if "attention_backend" in runtime:
        backend = runtime["attention_backend"]
        if isinstance(backend, str):
            try:
                backend = AttnBackend[backend]
            except KeyError as exc:
                raise ValueError(f"unsupported Spark attention backend: {backend}") from exc
        if not isinstance(backend, AttnBackend):
            raise ValueError(f"unsupported Spark attention backend: {backend}")
        runtime["attention_backend"] = backend
    dtype = params_dtype or getattr(torch, spark.parameter_dtype)
    if spark.moe_is_enabled and not spark.moe_router_in_fp32:
        raise ValueError("serial MoE requires moe_router_in_fp32=True for FP32 combine")
    config = MuseSpark1xTransformerConfig(
        **map_to_megatron_fields(spark),
        spark=spark,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        pipeline_dtype=torch.float32 if spark.residual_dtype == "fp32" else dtype,
        microbatch_group_size_per_vp_stage=1,
        qk_layernorm=spark.use_qk_norm,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        softmax_scale=spark.head_dim**-0.5,
        apply_rope_fusion=True,
        fused_single_qkv_rope=False,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        moe_router_load_balancing_type="none",
        moe_deterministic_combine=spark.moe_is_enabled,
        moe_grouped_gemm=spark.moe_is_enabled,
        **runtime,
    )
    validate_model_config(config)
    return config
