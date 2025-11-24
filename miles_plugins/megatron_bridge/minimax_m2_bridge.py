import logging

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.core.models.gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)


# ref: Qwen3MoEBridge
@MegatronModelBridge.register_bridge(source="MinimaxM2ForCausalLM", target=GPTModel)
class MinimaxM2Bridge(MegatronModelBridge):
    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Input layernorm - fused with TE (default layer spec)
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            # Input layernorm - separate (quantization layer spec)
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.block_sparse_moe.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.block_sparse_moe.e_score_correction_bias",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                # Note: Qwen3 MoE does NOT have bias in QKV projections
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Expert mappings for TEGroupedMLP
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.block_sparse_moe.experts.linear_fc1.weight*",
                    gate="model.layers.*.block_sparse_moe.experts.*.gate_proj.weight",
                    up="model.layers.*.block_sparse_moe.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.block_sparse_moe.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.block_sparse_moe.experts.*.down_proj.weight",
                ),
                # Expert mappings for SequentialMLP (used by quantization)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.block_sparse_moe.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.block_sparse_moe.experts.*.gate_proj.weight",
                    up="model.layers.*.block_sparse_moe.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.block_sparse_moe.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.block_sparse_moe.experts.*.down_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
