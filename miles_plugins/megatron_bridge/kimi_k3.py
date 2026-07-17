from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel

from miles_plugins.models.kimi_k3.model import build_kimi_k3_spec
from miles_plugins.models.kimi_k3.ops import situ_and_mul


@dataclass
class KimiK3ModelProvider(MLAModelProvider):
    kimi_kda_layers: tuple[int, ...] = ()
    kimi_linear_num_heads: int = 0
    kimi_linear_head_dim: int = 0
    kimi_linear_conv_kernel_size: int = 0
    kimi_kda_gate_lower_bound: float = -5.0
    kimi_attn_res_block_size: int = 0
    moe_latent_use_norm: bool = True
    gated_activation_func: Callable = situ_and_mul


_AUTO_MAPPINGS = {
    "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
    "output_layer.weight": "language_model.lm_head.weight",
    "decoder.layers.*.mlp.experts.linear_fc2.weight*": (
        "language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight"
    ),
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": (
        "language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight"
    ),
}

_COLUMN_PARALLEL_MAPPINGS = {
    "decoder.layers.*.self_attention.q_proj.weight": (
        "language_model.model.layers.*.self_attn.q_proj.weight"
    ),
    "decoder.layers.*.self_attention.k_proj.weight": (
        "language_model.model.layers.*.self_attn.k_proj.weight"
    ),
    "decoder.layers.*.self_attention.v_proj.weight": (
        "language_model.model.layers.*.self_attn.v_proj.weight"
    ),
    "decoder.layers.*.self_attention.q_conv1d.weight": (
        "language_model.model.layers.*.self_attn.q_conv1d.weight"
    ),
    "decoder.layers.*.self_attention.k_conv1d.weight": (
        "language_model.model.layers.*.self_attn.k_conv1d.weight"
    ),
    "decoder.layers.*.self_attention.v_conv1d.weight": (
        "language_model.model.layers.*.self_attn.v_conv1d.weight"
    ),
    "decoder.layers.*.self_attention.dt_bias": (
        "language_model.model.layers.*.self_attn.dt_bias"
    ),
    "decoder.layers.*.self_attention.f_b_proj.weight": (
        "language_model.model.layers.*.self_attn.f_b_proj.weight"
    ),
    "decoder.layers.*.self_attention.b_proj.weight": (
        "language_model.model.layers.*.self_attn.b_proj.weight"
    ),
    "decoder.layers.*.self_attention.g_proj.weight": (
        "language_model.model.layers.*.self_attn.g_proj.weight"
    ),
    "decoder.layers.*.self_attention.q_b_proj.weight": (
        "language_model.model.layers.*.self_attn.q_b_proj.weight"
    ),
    "decoder.layers.*.self_attention.kv_b_proj.weight": (
        "language_model.model.layers.*.self_attn.kv_b_proj.weight"
    ),
}

_ROW_PARALLEL_MAPPINGS = {
    "decoder.layers.*.self_attention.o_proj.weight": (
        "language_model.model.layers.*.self_attn.o_proj.weight"
    ),
    "decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
        "language_model.model.layers.*.block_sparse_moe.shared_experts.down_proj.weight"
    ),
}

_REPLICATED_MAPPINGS = {
    "decoder.final_layernorm.weight": "language_model.model.norm.weight",
    "decoder.layers.*.input_layernorm.weight": "language_model.model.layers.*.input_layernorm.weight",
    "decoder.layers.*.pre_mlp_layernorm.weight": (
        "language_model.model.layers.*.post_attention_layernorm.weight"
    ),
    "decoder.layers.*.self_attention.f_a_proj.weight": (
        "language_model.model.layers.*.self_attn.f_a_proj.weight"
    ),
    "decoder.layers.*.self_attention.o_norm.weight": (
        "language_model.model.layers.*.self_attn.o_norm.weight"
    ),
    "decoder.layers.*.self_attention.q_a_proj.weight": (
        "language_model.model.layers.*.self_attn.q_a_proj.weight"
    ),
    "decoder.layers.*.self_attention.q_a_layernorm.weight": (
        "language_model.model.layers.*.self_attn.q_a_layernorm.weight"
    ),
    "decoder.layers.*.self_attention.kv_a_proj_with_mqa.weight": (
        "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa.weight"
    ),
    "decoder.layers.*.self_attention.kv_a_layernorm.weight": (
        "language_model.model.layers.*.self_attn.kv_a_layernorm.weight"
    ),
    "decoder.layers.*.self_attention_res_norm.weight": (
        "language_model.model.layers.*.self_attention_res_norm.weight"
    ),
    "decoder.layers.*.self_attention_res_proj.weight": (
        "language_model.model.layers.*.self_attention_res_proj.weight"
    ),
    "decoder.layers.*.mlp_res_norm.weight": "language_model.model.layers.*.mlp_res_norm.weight",
    "decoder.layers.*.mlp_res_proj.weight": "language_model.model.layers.*.mlp_res_proj.weight",
    "decoder.layers.*.mlp.router.weight": (
        "language_model.model.layers.*.block_sparse_moe.gate.weight"
    ),
    "decoder.layers.*.mlp.router.expert_bias": (
        "language_model.model.layers.*.block_sparse_moe.gate.e_score_correction_bias"
    ),
    "decoder.layers.*.mlp.fc1_latent_proj.weight": (
        "language_model.model.layers.*.block_sparse_moe.routed_expert_down_proj.weight"
    ),
    "decoder.layers.*.mlp.routed_expert_norm.weight": (
        "language_model.model.layers.*.block_sparse_moe.routed_expert_norm.weight"
    ),
    "decoder.layers.*.mlp.fc2_latent_proj.weight": (
        "language_model.model.layers.*.block_sparse_moe.routed_expert_up_proj.weight"
    ),
}


class KimiK3ALogMapping(ColumnParallelMapping):
    def __init__(self, megatron_param: str, hf_param: str, num_heads: int):
        super().__init__(megatron_param, hf_param)
        self.num_heads = num_heads

    def resolve(self, captures: tuple[str, ...]):
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param, self.num_heads)

    def hf_to_megatron(self, hf_weights, megatron_module):
        if hf_weights is not None:
            assert hf_weights.ndim == 1
            assert hf_weights.shape[0] >= self.num_heads
            hf_weights = hf_weights[: self.num_heads].contiguous()
        return super().hf_to_megatron(hf_weights, megatron_module)


@MegatronModelBridge.register_bridge(
    source="KimiK3ForConditionalGeneration",
    target=GPTModel,
    provider=KimiK3ModelProvider,
    model_type="kimi_k3",
)
class KimiK3MegatronBridge(MegatronModelBridge):
    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        if hidden_act == "situ":
            return situ_and_mul
        return super().hf_to_megatron_activation(hidden_act)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> KimiK3ModelProvider:
        text_config = hf_pretrained.config.text_config
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider_kwargs.pop("_mla_rope_params", None)
        valid_fields = KimiK3ModelProvider.__dataclass_fields__
        provider = KimiK3ModelProvider(**{key: value for key, value in provider_kwargs.items() if key in valid_fields})

        provider.transformer_layer_spec = build_kimi_k3_spec
        provider.position_embedding_type = "none"
        provider.normalization = "RMSNorm"
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.gated_linear_unit = True
        provider.gated_activation_func = situ_and_mul
        provider.bias_activation_fusion = False
        provider.bias_dropout_fusion = False
        provider.use_te_activation_func = False
        provider.persist_layer_norm = True
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0

        provider.num_moe_experts = text_config.num_experts
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
        provider.moe_router_topk = text_config.num_experts_per_token
        provider.moe_router_score_function = text_config.moe_router_activation_func
        provider.moe_router_topk_scaling_factor = text_config.routed_scaling_factor
        provider.moe_router_num_groups = text_config.num_expert_group
        provider.moe_router_group_topk = text_config.topk_group
        provider.moe_router_pre_softmax = True
        provider.moe_router_enable_expert_bias = True
        provider.freeze_e_score_correction_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_shared_expert_overlap = False
        provider.moe_shared_expert_intermediate_size = (
            text_config.moe_intermediate_size * text_config.num_shared_experts
        )
        provider.moe_latent_size = text_config.routed_expert_hidden_size
        provider.moe_latent_use_norm = text_config.latent_moe_use_norm
        provider.moe_layer_freq = [
            int(
                layer_idx >= text_config.first_k_dense_replace
                and layer_idx % text_config.moe_layer_freq == 0
            )
            for layer_idx in range(text_config.num_hidden_layers)
        ]

        linear_config = text_config.linear_attn_config
        provider.kimi_kda_layers = tuple(linear_config["kda_layers"])
        provider.kimi_linear_num_heads = linear_config["num_heads"]
        provider.kimi_linear_head_dim = linear_config["head_dim"]
        provider.kimi_linear_conv_kernel_size = linear_config["short_conv_kernel_size"]
        provider.kimi_kda_gate_lower_bound = linear_config["gate_lower_bound"]
        provider.kimi_attn_res_block_size = text_config.attn_res_block_size
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in _AUTO_MAPPINGS.items()
        ]
        mappings.extend(
            ColumnParallelMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in _COLUMN_PARALLEL_MAPPINGS.items()
        )
        mappings.extend(
            RowParallelMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in _ROW_PARALLEL_MAPPINGS.items()
        )
        mappings.extend(
            ReplicatedMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in _REPLICATED_MAPPINGS.items()
        )
        mappings.append(
            KimiK3ALogMapping(
                megatron_param="decoder.layers.*.self_attention.A_log",
                hf_param="language_model.model.layers.*.self_attn.A_log",
                num_heads=self.hf_config.text_config.linear_attn_config["num_heads"],
            )
        )
        mappings.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate=(
                        "language_model.model.layers.*.block_sparse_moe.shared_experts.gate_proj.weight"
                    ),
                    up=(
                        "language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj.weight"
                    ),
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    up="language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    up="language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
            ]
        )

        last_layer = self.hf_config.text_config.num_hidden_layers - 1
        mappings.extend(
            [
                ReplicatedMapping(
                    megatron_param=f"decoder.layers.{last_layer}.output_attn_res_norm.weight",
                    hf_param="language_model.model.output_attn_res_norm.weight",
                ),
                ReplicatedMapping(
                    megatron_param=f"decoder.layers.{last_layer}.output_attn_res_proj.weight",
                    hf_param="language_model.model.output_attn_res_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)
