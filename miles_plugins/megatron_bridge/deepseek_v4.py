"""Conversion-only Megatron-Bridge support for Miles' native DeepSeek-V4."""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel


_DSV4_ATTENTION_MAPPINGS = {
    "decoder.layers.*.self_attention.wq_a.weight": "model.layers.*.self_attn.wq_a.weight",
    "decoder.layers.*.self_attention.q_norm.weight": "model.layers.*.self_attn.q_norm.weight",
    "decoder.layers.*.self_attention.wq_b.weight": "model.layers.*.self_attn.wq_b.weight",
    "decoder.layers.*.self_attention.wkv.weight": "model.layers.*.self_attn.wkv.weight",
    "decoder.layers.*.self_attention.kv_norm.weight": "model.layers.*.self_attn.kv_norm.weight",
    "decoder.layers.*.self_attention.wo_a.weight": "model.layers.*.self_attn.wo_a.weight",
    "decoder.layers.*.self_attention.wo_b.weight": "model.layers.*.self_attn.wo_b.weight",
}

_DSV4_REPLICATED_MAPPINGS = {
    "decoder.layers.*.hc_attn_fn": "model.layers.*.hc_attn_fn",
    "decoder.layers.*.hc_attn_base": "model.layers.*.hc_attn_base",
    "decoder.layers.*.hc_attn_scale": "model.layers.*.hc_attn_scale",
    "decoder.layers.*.hc_ffn_fn": "model.layers.*.hc_ffn_fn",
    "decoder.layers.*.hc_ffn_base": "model.layers.*.hc_ffn_base",
    "decoder.layers.*.hc_ffn_scale": "model.layers.*.hc_ffn_scale",
    "decoder.layers.*.self_attention.compressor.ape": "model.layers.*.self_attn.compressor.ape",
    "decoder.layers.*.self_attention.compressor.wkv.weight": "model.layers.*.self_attn.compressor.wkv.weight",
    "decoder.layers.*.self_attention.compressor.wgate.weight": "model.layers.*.self_attn.compressor.wgate.weight",
    "decoder.layers.*.self_attention.compressor.norm.weight": "model.layers.*.self_attn.compressor.norm.weight",
    "decoder.layers.*.self_attention.indexer.compressor.ape": "model.layers.*.self_attn.indexer.compressor.ape",
    "decoder.layers.*.self_attention.indexer.compressor.wkv.weight": (
        "model.layers.*.self_attn.indexer.compressor.wkv.weight"
    ),
    "decoder.layers.*.self_attention.indexer.compressor.wgate.weight": (
        "model.layers.*.self_attn.indexer.compressor.wgate.weight"
    ),
    "decoder.layers.*.self_attention.indexer.compressor.norm.weight": (
        "model.layers.*.self_attn.indexer.compressor.norm.weight"
    ),
    "decoder.layers.*.mlp.router.tid2eid": "model.layers.*.mlp.topk.tid2eid",
    "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
    "decoder.hc_head_params.hc_head_fn": "model.hc_head_fn",
    "decoder.hc_head_params.hc_head_base": "model.hc_head_base",
    "decoder.hc_head_params.hc_head_scale": "model.hc_head_scale",
}

_DSV4_COLUMN_PARALLEL_MAPPINGS = {
    "decoder.layers.*.self_attention.attn_sink": "model.layers.*.self_attn.attn_sink",
}

_DSV4_AUTO_MAPPINGS = {
    "decoder.layers.*.self_attention.indexer.linear_wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
    "decoder.layers.*.self_attention.indexer.linear_weights_proj.weight": (
        "model.layers.*.self_attn.indexer.weights_proj.weight"
    ),
}


def is_deepseek_v4_config(hf_config) -> bool:
    architectures = getattr(hf_config, "architectures", None) or []
    return bool(architectures and architectures[0] == "DeepseekV4ForCausalLM")


def _get_dsv4_explicit_mappings():
    mappings = [
        AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
        for megatron_param, hf_param in _DSV4_ATTENTION_MAPPINGS.items()
    ]
    mappings.extend(
        ReplicatedMapping(megatron_param=megatron_param, hf_param=hf_param)
        for megatron_param, hf_param in _DSV4_REPLICATED_MAPPINGS.items()
    )
    mappings.extend(
        ColumnParallelMapping(megatron_param=megatron_param, hf_param=hf_param)
        for megatron_param, hf_param in _DSV4_COLUMN_PARALLEL_MAPPINGS.items()
    )
    mappings.extend(
        AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
        for megatron_param, hf_param in _DSV4_AUTO_MAPPINGS.items()
    )
    return mappings


@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class MilesDeepSeekV4Bridge(DeepSeekV3Bridge):
    """Map the native V4 graph for adapter publication and export only."""

    def provider_bridge(self, hf_pretrained):
        raise RuntimeError(
            "DeepSeek-V4 model construction must use Miles' native dsv4 provider; "
            "this bridge is conversion-only"
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = get_common_mapping_list(hf_config=self.hf_config)
        mappings.extend(_get_dsv4_explicit_mappings())
        return MegatronMappingRegistry(*mappings)


__all__ = ["MilesDeepSeekV4Bridge", "is_deepseek_v4_config"]
