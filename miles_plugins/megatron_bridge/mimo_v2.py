"""Megatron-Bridge support for XiaomiMiMo MiMo-V2 (``MiMoV2ForCausalLM``, e.g. MiMo-V2.6-Flash-RL).

Builds on NVIDIA Megatron-Bridge ``megatron/bridge/models/mimo_v2_flash``, which covers the earlier
MiMo-V2-Flash checkpoint, and reuses its q/k/v weight mapping. MiMo-V2 attention differs from the
stock GPT layer in four ways:

* per-layer KV heads: SWA layers use ``swa_num_key_value_heads``, global layers ``num_key_value_heads``;
* a learnable attention sink on SWA layers only, while ``softmax_type`` is one global setting;
* asymmetric head dims: Q/K use ``head_dim`` (192) and V uses ``v_head_dim`` (128);
* V is multiplied by ``attention_value_scale`` before attention.

The first two are per-layer config values: the provider sets ``heterogeneous_block_specs``, so
``TransformerBlock`` builds each layer from ``MiMoV2ModelProvider.get_config_for_layer``. The other
two need ``MiMoV2SelfAttention`` and ``MiMoV2TEDotProductAttention``. The per-layer RoPE base and the
SWA window reuse existing config fields (``rotary_base_per_layer``, ``window_size`` with a per-layer
``window_attn_skip_freq``) instead of Megatron-Bridge's dual-base RoPE and window rule.

The bridge reads the BF16 split-q/k/v layout written by ``tools/convert_mimo_v2_to_bf16.py``; the
official checkpoint (FP8/MXFP4, fused kv-head-interleaved ``qkv_proj``) must be converted first.
Only the text decoder is built: vision/audio towers and the MTP layers are not.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashQKVMapping
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import SelfAttention

logger = logging.getLogger(__name__)


def _sized_linear(build, *, input_size: int | None = None, output_size: int | None = None):
    """``build`` with the given input/output size in place of the one SelfAttention passes."""

    def build_sized(base_input_size, base_output_size, **kwargs):
        return build(
            base_input_size if input_size is None else input_size,
            base_output_size if output_size is None else output_size,
            **kwargs,
        )

    return build_sized


class MiMoV2SelfAttention(SelfAttention):
    """SelfAttention whose V heads (``v_head_dim``) are narrower than its Q/K heads (``kv_channels``).

    The stock layer sizes ``linear_qkv``, ``linear_proj`` and the q/k/v split with ``kv_channels`` for V;
    this builds the two projections once at the MiMo sizes and splits accordingly. The layer config
    already carries this layer's KV heads (``MiMoV2ModelProvider.get_config_for_layer``).
    """

    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        assert not (config.attention_output_gate or config.head_wise_attn_gate), "MiMo-V2 has no attention gate"
        # qk_clip reads linear_qkv as q + 2 * kv rows of kv_channels each, which does not hold for a narrower V.
        assert not config.qk_clip, "qk_clip does not support a V head narrower than Q/K"
        heads, groups = config.num_attention_heads, config.num_query_groups
        qkv_out_dim = heads * config.kv_channels + groups * (config.kv_channels + config.v_head_dim)
        submodules = dataclasses.replace(
            submodules,
            linear_qkv=_sized_linear(submodules.linear_qkv, output_size=qkv_out_dim),
            linear_proj=_sized_linear(submodules.linear_proj, input_size=heads * config.v_head_dim),
        )
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        assert groups >= self.world_size, "TP must not exceed the layer's KV heads"
        self.linear_qkv_out_dim = qkv_out_dim
        self.val_hidden_size = config.v_head_dim

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, head_wise_gate=False, split_qkv=True
    ):
        assert not (output_gate or head_wise_gate) and split_qkv, "MiMo-V2 attention needs the split q/k/v path"
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        groups = self.num_query_groups_per_partition
        heads_per_group = self.num_attention_heads_per_partition // groups
        qk_dim, v_dim = self.hidden_size_per_attention_head, self.config.v_head_dim
        # [sq, b, ng * (np/ng * hn + hn + hv)] -> [sq, b, ng, np/ng * hn + hn + hv]
        mixed_qkv = mixed_qkv.view(*mixed_qkv.shape[:-1], groups, heads_per_group * qk_dim + qk_dim + v_dim)
        query, key, value = torch.split(mixed_qkv, [heads_per_group * qk_dim, qk_dim, v_dim], dim=3)
        query = query.reshape(query.size(0), query.size(1), -1, qk_dim)
        return query, key, value


class MiMoV2TEDotProductAttention(TEDotProductAttention):
    """TE core attention with separate Q/K and V channels and V scaled by ``attention_value_scale``.

    The learnable sink comes from the layer config's ``softmax_type``.
    """

    def __init__(self, config, layer_number, attn_mask_type, attention_type, attention_dropout=None, **kwargs):
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            k_channels=config.kv_channels,
            v_channels=config.v_head_dim,
            **kwargs,
        )
        self.attention_value_scale = config.attention_value_scale

    def forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
        if self.attention_value_scale is not None:
            value = value * self.attention_value_scale
        return super().forward(query, key, value, attention_mask, attn_mask_type, **kwargs)


AutoMapping.register_module_type("MiMoV2TEDotProductAttention", "column")


def mimo_v2_layer_spec(config, vp_stage: int | None = None) -> ModuleSpec:
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    for layer_spec in spec.layer_specs:
        layer_spec.submodules.self_attention.module = MiMoV2SelfAttention
        layer_spec.submodules.self_attention.submodules.core_attention = MiMoV2TEDotProductAttention
    return spec


@dataclass
class MiMoV2ModelProvider(GPTModelProvider):
    """GPT provider plus the MiMo-V2 attention fields; each layer is built from ``get_config_for_layer``."""

    transformer_layer_spec: ModuleSpec | Callable = field(default_factory=lambda: mimo_v2_layer_spec)
    # Makes TransformerBlock build each layer from get_config_for_layer.
    heterogeneous_block_specs: bool = True
    hybrid_attention_pattern: list[int] | None = None
    full_attn_num_query_groups: int = 4
    swa_num_query_groups: int = 8
    v_head_dim: int = 128
    attention_value_scale: float | None = None
    full_attention_sink: bool = False
    swa_attention_sink: bool = True

    def get_config_for_layer(self, layer_number: int) -> MiMoV2ModelProvider:
        """Shallow copy with the KV heads and attention sink of global layer ``layer_number`` (1-indexed)."""
        assert layer_number <= len(self.hybrid_attention_pattern), f"no attention pattern for layer {layer_number}"
        swa = self.hybrid_attention_pattern[layer_number - 1]
        config = copy.copy(self)
        config.num_query_groups = self.swa_num_query_groups if swa else self.full_attn_num_query_groups
        sink = self.swa_attention_sink if swa else self.full_attention_sink
        config.softmax_type = "learnable" if sink else "vanilla"
        return config

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        assert self.tensor_model_parallel_size <= min(
            self.full_attn_num_query_groups, self.swa_num_query_groups
        ), "MiMo-V2 needs TP <= the smallest per-layer KV head count"
        assert self.context_parallel_size == 1, "MiMo-V2 attention does not support context parallelism yet"
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@MegatronModelBridge.register_bridge(
    source="MiMoV2ForCausalLM", target=GPTModel, provider=MiMoV2ModelProvider, model_type="mimo_v2"
)
class MiMoV2Bridge(MegatronModelBridge):
    """HF ``MiMoV2ForCausalLM`` (BF16, split q/k/v) <-> Megatron GPTModel with MiMo-V2 attention."""

    def provider_bridge(self, hf_pretrained) -> MiMoV2ModelProvider:
        hf = hf_pretrained.config
        if getattr(hf, "attention_projection_layout", "split") != "split" or getattr(hf, "quantization_config", None):
            raise ValueError(
                "MiMo-V2 bridge reads BF16 split q/k/v checkpoints; convert the official checkpoint "
                "with tools/convert_mimo_v2_to_bf16.py first"
            )
        for swa_key, full_key in (
            ("swa_num_attention_heads", "num_attention_heads"),
            ("swa_head_dim", "head_dim"),
            ("swa_v_head_dim", "v_head_dim"),
        ):
            assert getattr(hf, swa_key) == getattr(hf, full_key), f"{swa_key} must equal {full_key}"

        provider = super().provider_bridge(hf_pretrained)
        pattern = [int(p) for p in hf.hybrid_layer_pattern]
        provider.hybrid_attention_pattern = pattern
        # A SWA layer attends the query plus sliding_window_size previous keys, like SGLang's
        # RadixAttention (whose window excludes the query token), so rollout and training agree.
        provider.window_size = (hf.sliding_window_size, 0)
        provider.window_attn_skip_freq = pattern
        provider.rotary_base = hf.rope_theta
        provider.rotary_base_per_layer = [hf.swa_rope_theta if p else hf.rope_theta for p in pattern]
        provider.apply_rope_fusion = False
        provider.full_attn_num_query_groups = hf.num_key_value_heads
        provider.swa_num_query_groups = hf.swa_num_key_value_heads
        provider.num_query_groups = hf.num_key_value_heads
        provider.v_head_dim = hf.v_head_dim
        provider.attention_value_scale = hf.attention_value_scale
        provider.full_attention_sink = bool(hf.add_full_attention_sink_bias)
        provider.swa_attention_sink = bool(hf.add_swa_attention_sink_bias)

        provider.normalization = "RMSNorm"
        provider.layernorm_epsilon = hf.layernorm_epsilon
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.position_embedding_type = "rope"
        provider.share_embeddings_and_output_weights = False
        # The HF model has no hidden dropout; the TransformerConfig default (0.1) would stay on, since
        # Miles' bridge mode does not copy --hidden-dropout onto the provider (the bundled bridges
        # set it the same way). attention_dropout comes from the HF config (0.0).
        provider.hidden_dropout = 0.0

        provider.moe_layer_freq = [int(f) for f in hf.moe_layer_freq]
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_router_dtype = "fp32"
        if getattr(hf, "n_group", None) in (None, 1):
            provider.moe_router_num_groups = None
            provider.moe_router_group_topk = None
        provider.mtp_num_layers = None
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        direct = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.core_attention.softmax_offset": "model.layers.*.self_attn.attention_sink_bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "model.layers.*.mlp.experts.*.down_proj.weight",
        }
        gated = {
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp",
            "decoder.layers.*.mlp.experts.linear_fc1.weight*": "model.layers.*.mlp.experts.*",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": "model.layers.*.mlp.experts.*",
        }
        mappings = [AutoMapping(megatron_param=m, hf_param=h) for m, h in direct.items()]
        mappings += [
            GatedMLPMapping(megatron_param=m, gate=f"{h}.gate_proj.weight", up=f"{h}.up_proj.weight")
            for m, h in gated.items()
        ]
        mappings.append(
            MiMoV2FlashQKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )
        return MegatronMappingRegistry(*mappings)
