from functools import partial

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec

from miles_plugins.models.muse_spark.ops.norms import MuseSpark1xRMSNorm


class MuseSpark1xDotProductAttention(TEDotProductAttention):
    def __init__(self, config, *args, **kwargs):
        scale = kwargs.pop("softmax_scale", None)
        if scale is None:
            scale = config.softmax_scale
        scale = config.kv_channels**-0.5 if scale is None else scale
        if config.spark.use_qk_norm:
            scale *= config.spark.query_multiplier
        super().__init__(config, *args, softmax_scale=scale, **kwargs)
        self.spark = config.spark
        self.output_norm = torch.nn.RMSNorm(config.kv_channels, eps=self.spark.rms_norm_eps, elementwise_affine=False)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=AttnMaskType.causal,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs
    ):
        output = super().forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            **kwargs
        )
        if self.spark.use_attn_o_norm and not self.spark.use_attn_o_gate:
            shape = output.shape
            output = self.output_norm(output.reshape(*shape[:-1], -1, self.spark.head_dim))
            output = output.reshape(shape)
        return output


class MuseSpark1xSelfAttention(SelfAttention):
    def __init__(self, config, submodules, layer_number, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        self.output_norm = torch.nn.RMSNorm(
            config.kv_channels,
            eps=config.spark.rms_norm_eps,
            elementwise_affine=False,
        )

    def _apply_output_gate(self, x, gate):
        value = x.float()
        if self.config.spark.use_attn_o_norm:
            value = self.output_norm(value.reshape(*x.shape[:-1], -1, self.config.kv_channels)).reshape(x.shape)
        return (value * torch.sigmoid(gate.reshape(x.shape).float())).to(x.dtype)


def get_muse_spark_1x_attention_spec(config):
    norm = partial(MuseSpark1xRMSNorm, has_weight=False) if config.spark.use_qk_norm else None
    return ModuleSpec(
        module=MuseSpark1xSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=TEColumnParallelLinear,
            linear_proj=TERowParallelLinear,
            core_attention=MuseSpark1xDotProductAttention,
            q_layernorm=norm,
            k_layernorm=norm,
        ),
    )
