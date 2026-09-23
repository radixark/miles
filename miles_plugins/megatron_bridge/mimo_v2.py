"""Megatron-Bridge support for XiaomiMiMo MiMo-V2 (``MiMoV2ForCausalLM``, e.g. MiMo-V2.6-Flash-RL).

Adapted from NVIDIA Megatron-Bridge ``megatron/bridge/models/mimo_v2_flash`` (Apache-2.0), which
covers the earlier MiMo-V2-Flash checkpoint. MiMo-V2 attention differs from the stock GPT layer
in four ways that the TransformerConfig cannot express on its own:

* per-layer KV heads: SWA layers use ``swa_num_key_value_heads``, global layers ``num_key_value_heads``;
* asymmetric head dims: Q/K use ``head_dim`` (192) and V uses ``v_head_dim`` (128);
* a learnable attention sink on SWA layers only, while ``softmax_type`` is one global setting;
* V is multiplied by ``attention_value_scale`` before attention.

The per-layer RoPE base and the SWA window reuse existing config fields (``rotary_base_per_layer``,
``window_size`` with a per-layer ``window_attn_skip_freq``).

The bridge reads the BF16 split-q/k/v layout written by ``tools/convert_mimo_v2_to_bf16.py``; the
official checkpoint (FP8/MXFP4, fused kv-head-interleaved ``qkv_proj``) must be converted first.
Only the text decoder is built: vision/audio towers and the MTP layers are not.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.conversion.utils import remove_non_pickleables
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import SelfAttention

logger = logging.getLogger(__name__)


def _layer_config(config, layer_number: int):
    """Shallow config copy carrying the attention shape and softmax of this 1-indexed layer."""
    assert layer_number <= len(config.hybrid_attention_pattern), f"no attention pattern for layer {layer_number}"
    config = copy.copy(config)
    if config.hybrid_attention_pattern[layer_number - 1]:
        config.num_query_groups = config.swa_num_query_groups
        config.softmax_type = "learnable" if config.swa_attention_sink else "vanilla"
    else:
        config.num_query_groups = config.full_attn_num_query_groups
        config.softmax_type = "learnable" if config.full_attention_sink else "vanilla"
    return config


class MiMoV2SelfAttention(SelfAttention):
    """SelfAttention with per-layer KV heads and a V head narrower than Q/K."""

    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        config = _layer_config(config, layer_number)
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        assert not (config.attention_output_gate or config.head_wise_attn_gate), "MiMo-V2 has no attention gate"
        assert config.num_query_groups >= self.world_size, "TP must not exceed the layer's KV heads"
        name = kwargs.get("name")
        self.val_hidden_size = config.v_head_dim
        self.linear_qkv_out_dim = self.query_projection_size + config.num_query_groups * (
            config.kv_channels + config.v_head_dim
        )
        self.linear_qkv = submodules.linear_qkv(
            config.hidden_size,
            self.linear_qkv_out_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
            tp_group=self.pg_collection.tp,
            name=(name + ".linear_qkv") if name is not None else None,
        )
        self.linear_proj = submodules.linear_proj(
            config.v_head_dim * config.num_attention_heads,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
            name=(name + ".linear_proj") if name is not None else None,
        )

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
    """TE core attention with this layer's sink setting, Q/K vs V channels and V scaling."""

    def __init__(self, config, layer_number, attn_mask_type, attention_type, attention_dropout=None, **kwargs):
        config = _layer_config(config, layer_number)
        kwargs["k_channels"] = config.kv_channels
        kwargs["v_channels"] = config.v_head_dim
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
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
    """GPT provider plus the per-layer attention fields read by the MiMo-V2 modules."""

    transformer_layer_spec: ModuleSpec | Callable = field(default_factory=lambda: mimo_v2_layer_spec)
    hybrid_attention_pattern: list[int] | None = None
    full_attn_num_query_groups: int = 4
    swa_num_query_groups: int = 8
    v_head_dim: int = 128
    attention_value_scale: float | None = None
    full_attention_sink: bool = False
    swa_attention_sink: bool = True

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        assert self.tensor_model_parallel_size <= min(
            self.full_attn_num_query_groups, self.swa_num_query_groups
        ), "MiMo-V2 needs TP <= the smallest per-layer KV head count"
        assert self.context_parallel_size == 1, "MiMo-V2 attention does not support context parallelism yet"
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


class MiMoV2QKVMapping(QKVMapping):
    """QKV mapping for per-group [q heads, k head, v head] with a V head of ``v_head_dim``."""

    def hf_to_megatron(self, hf_weights, megatron_module):
        merged = None
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            q, k, v = hf_weights["q"], hf_weights["k"], hf_weights["v"]
            assert q.ndim == 2, "MiMo-V2 has no qkv bias"
            groups, qk_dim, v_dim = config.num_query_groups, config.kv_channels, config.v_head_dim
            q = q.view(groups, -1, q.shape[-1])
            merged = torch.cat([q, k.view(groups, qk_dim, -1), v.view(groups, v_dim, -1)], dim=1).flatten(0, 1)
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, "qkv_config")
        else:
            config = remove_non_pickleables(self._get_config(megatron_module), max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, "qkv_config")
        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}
        packed = next(iter(packed_dict.values()))
        groups, qk_dim, v_dim = config.num_query_groups, config.kv_channels, config.v_head_dim
        q_dim = config.num_attention_heads // groups * qk_dim
        q, k, v = packed.view(groups, q_dim + qk_dim + v_dim, -1).split([q_dim, qk_dim, v_dim], dim=1)
        return {
            self.hf_param["q"]: q.reshape(-1, packed.shape[-1]),
            self.hf_param["k"]: k.reshape(-1, packed.shape[-1]),
            self.hf_param["v"]: v.reshape(-1, packed.shape[-1]),
        }


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
        provider.window_size = (hf.sliding_window_size - 1, 0)
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
            MiMoV2QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )
        return MegatronMappingRegistry(*mappings)
