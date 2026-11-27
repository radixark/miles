from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from miles_plugins.models.muse_spark.ops.attention import get_muse_spark_1x_attention_spec
from miles_plugins.models.muse_spark.ops.moe import get_muse_spark_1x_moe_spec
from miles_plugins.models.muse_spark.ops.norms import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    calculate_gain_center,
)
from miles_plugins.models.muse_spark.ops.residual import MuseSpark1xBiasDropoutAdd


class MuseSpark1xNormFactory:
    """Use the model's epsilon instead of the shared epsilon supplied by MCore."""

    def __new__(cls, config, hidden_size, eps, *, norm_type, norm_eps, **kwargs):
        return norm_type(hidden_size, eps=norm_eps, config=config, **kwargs)


class MuseSpark1xDenseMLP(MLP):
    def __init__(self, config, pg_collection, is_mtp_layer=False, layer_number=None, name=None):
        super().__init__(
            config,
            MLPSubmodules(TEColumnParallelLinear, TERowParallelLinear),
            ffn_hidden_size=config.ffn_hidden_size,
            tp_group=pg_collection.tp,
            name=name,
        )

    def forward(self, hidden_states, *args, **kwargs):
        return super().forward(hidden_states.to(self.config.params_dtype), *args, **kwargs)


def get_muse_spark_1x_spec(config, pp_rank=None):
    spark = config.spark
    layer_specs = []
    layer_offset = get_transformer_layer_offset(config, pp_rank=pp_rank)
    num_layers = get_num_layers_to_build(config, pp_rank=pp_rank)

    def build_norm_spec(norm_type=MuseSpark1xRMSNorm, *, gain_center=1.0, **params):
        return ModuleSpec(
            MuseSpark1xNormFactory,
            params={
                "norm_type": norm_type,
                "norm_eps": spark.rms_norm_eps,
                "has_weight": spark.norm_affine,
                "weight_offset": gain_center if spark.norm_zero_centered_gamma else 0.0,
                **params,
            },
        )

    ffn_post_norm_type = MuseSpark1xRMSNormWithInputScale if spark.use_ffn_learnable_scales else MuseSpark1xRMSNorm

    residual_spec = ModuleSpec(
        MuseSpark1xBiasDropoutAdd,
        params={
            "dim": spark.hidden_size,
            "gated": spark.use_residual_gate,
            "dtype": config.params_dtype,
            "sequence_parallel": config.sequence_parallel,
        },
    )
    for layer_index in range(layer_offset, layer_offset + num_layers):
        layer_specs.append(
            ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=build_norm_spec(output_dtype=config.params_dtype),
                    self_attention=get_muse_spark_1x_attention_spec(config),
                    post_self_attn_layernorm=build_norm_spec(
                        norm_eps=spark.rms_post_norm_eps,
                        has_weight=spark.post_norm_affine,
                        gain_center=calculate_gain_center(spark.post_norm_gain_center_type, 2 * layer_index + 1),
                    ),
                    self_attn_bda=residual_spec,
                    pre_mlp_layernorm=build_norm_spec(
                        input_dtype=config.params_dtype,
                        output_dtype=config.params_dtype,
                    ),
                    mlp=get_muse_spark_1x_moe_spec(config) if spark.layer_is_moe(layer_index) else MuseSpark1xDenseMLP,
                    post_mlp_layernorm=build_norm_spec(
                        ffn_post_norm_type,
                        norm_eps=spark.rms_post_norm_eps,
                        has_weight=spark.post_norm_affine or spark.use_ffn_learnable_scales,
                        gain_center=calculate_gain_center(spark.post_norm_gain_center_type, 2 * layer_index + 2),
                    ),
                    mlp_bda=residual_spec,
                ),
            )
        )
    return TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=build_norm_spec(
            gain_center=calculate_gain_center(spark.output_norm_gain_center_type, -1),
            input_dtype=None if spark.output_norm_in_fp32 else config.params_dtype,
            output_dtype=config.params_dtype,
        ),
    )
