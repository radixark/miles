import inspect

import torch
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models import DeepseekV3Bridge

from miles_plugins.models.kimi_k3.model import build_kimi_k3_spec
from miles_plugins.models.kimi_k3.ops import situ_and_mul


@register_model("kimi_k3")
class KimiK3Bridge(DeepseekV3Bridge):
    TransformerConfigClass = MLATransformerConfig

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": ("attention_dropout", 0.0),
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": "head_dim",
    }

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "language_model.model.norm.weight",
        "output_layer.weight": "language_model.lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "input_layernorm.weight": ["language_model.model.layers.{layer_number}.input_layernorm.weight"],
        "self_attention.q_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.q_proj.weight"],
        "self_attention.k_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.k_proj.weight"],
        "self_attention.v_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.v_proj.weight"],
        "self_attention.q_conv1d.weight": ["language_model.model.layers.{layer_number}.self_attn.q_conv1d.weight"],
        "self_attention.k_conv1d.weight": ["language_model.model.layers.{layer_number}.self_attn.k_conv1d.weight"],
        "self_attention.v_conv1d.weight": ["language_model.model.layers.{layer_number}.self_attn.v_conv1d.weight"],
        "self_attention.A_log": ["language_model.model.layers.{layer_number}.self_attn.A_log"],
        "self_attention.dt_bias": ["language_model.model.layers.{layer_number}.self_attn.dt_bias"],
        "self_attention.f_a_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.f_a_proj.weight"],
        "self_attention.f_b_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.f_b_proj.weight"],
        "self_attention.b_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.b_proj.weight"],
        "self_attention.g_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.g_proj.weight"],
        "self_attention.o_norm.weight": ["language_model.model.layers.{layer_number}.self_attn.o_norm.weight"],
        "self_attention.o_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.q_a_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.q_a_proj.weight"],
        "self_attention.q_a_layernorm.weight": [
            "language_model.model.layers.{layer_number}.self_attn.q_a_layernorm.weight"
        ],
        "self_attention.q_b_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.q_b_proj.weight"],
        "self_attention.kv_a_proj_with_mqa.weight": [
            "language_model.model.layers.{layer_number}.self_attn.kv_a_proj_with_mqa.weight"
        ],
        "self_attention.kv_a_layernorm.weight": [
            "language_model.model.layers.{layer_number}.self_attn.kv_a_layernorm.weight"
        ],
        "self_attention.kv_b_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.kv_b_proj.weight"],
        "self_attention_res_norm.weight": [
            "language_model.model.layers.{layer_number}.self_attention_res_norm.weight"
        ],
        "self_attention_res_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attention_res_proj.weight"
        ],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["language_model.model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp_res_norm.weight": ["language_model.model.layers.{layer_number}.mlp_res_norm.weight"],
        "mlp_res_proj.weight": ["language_model.model.layers.{layer_number}.mlp_res_proj.weight"],
        "mlp.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.mlp.gate_proj.weight",
            "language_model.model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": ["language_model.model.layers.{layer_number}.mlp.down_proj.weight"],
        "mlp.router.weight": ["language_model.model.layers.{layer_number}.block_sparse_moe.gate.weight"],
        "mlp.router.expert_bias": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.gate.e_score_correction_bias"
        ],
        "mlp.fc1_latent_proj.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.routed_expert_down_proj.weight"
        ],
        "mlp.routed_expert_norm.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.routed_expert_norm.weight"
        ],
        "mlp.fc2_latent_proj.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.routed_expert_up_proj.weight"
        ],
        "mlp.shared_experts.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.shared_experts.gate_proj.weight",
            "language_model.model.layers.{layer_number}.block_sparse_moe.shared_experts.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.shared_experts.down_proj.weight"
        ],
        "mlp.experts.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight",
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight"
        ],
    }

    _OTHER_MAPPING = {
        "output_attn_res_norm.weight": ["language_model.model.output_attn_res_norm.weight"],
        "output_attn_res_proj.weight": ["language_model.model.output_attn_res_proj.weight"],
    }

    @property
    def text_config(self):
        return self.hf_config.text_config

    def _build_config(self):
        hf_config = self.text_config
        moe_layer_freq = [0] * hf_config.num_hidden_layers
        for layer_idx in range(hf_config.first_k_dense_replace, hf_config.num_hidden_layers):
            if layer_idx % hf_config.moe_layer_freq == 0:
                moe_layer_freq[layer_idx] = 1

        config = self._build_base_config(
            text_config_key="text_config",
            attention_backend=AttnBackend.auto,
            multi_latent_attention=True,
            qk_layernorm=True,
            q_lora_rank=hf_config.q_lora_rank,
            kv_lora_rank=hf_config.kv_lora_rank,
            qk_head_dim=hf_config.qk_nope_head_dim,
            qk_pos_emb_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            gated_activation_func=situ_and_mul,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            use_te_activation_func=False,
            persist_layer_norm=True,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            moe_latent_size=hf_config.routed_expert_hidden_size,
            moe_latent_use_norm=hf_config.latent_moe_use_norm,
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_token,
            moe_router_score_function=hf_config.moe_router_activation_func,
            moe_router_pre_softmax=True,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            moe_router_enable_expert_bias=True,
            freeze_e_score_correction_bias=True,
            moe_router_bias_update_rate=0.0,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_shared_expert_intermediate_size=(hf_config.moe_intermediate_size * hf_config.num_shared_experts),
            moe_shared_expert_overlap=False,
            moe_layer_freq=moe_layer_freq,
            disable_bf16_reduced_precision_matmul=True,
        )
        config.kimi_kda_layers = tuple(hf_config.linear_attn_config["kda_layers"])
        config.kimi_linear_num_heads = hf_config.linear_attn_config["num_heads"]
        config.kimi_linear_head_dim = hf_config.linear_attn_config["head_dim"]
        config.kimi_linear_conv_kernel_size = hf_config.linear_attn_config["short_conv_kernel_size"]
        config.kimi_kda_gate_lower_bound = hf_config.linear_attn_config["gate_lower_bound"]
        config.kimi_attn_res_block_size = hf_config.attn_res_block_size
        return config

    def _get_gptmodel_args(self) -> dict:
        return {
            "vocab_size": self.text_config.vocab_size,
            "max_sequence_length": self.text_config.max_position_embeddings,
            "position_embedding_type": "none",
        }

    def _get_transformer_layer_spec(self, vp_stage=None):
        self.has_vp_stage = "vp_stage" in inspect.signature(build_kimi_k3_spec).parameters
        return build_kimi_k3_spec(self.config, vp_stage=vp_stage)

    def _get_safetensor_io(self, weights_path: str):
        safetensor_io = SafeTensorIO(self._get_actual_hf_path(weights_path))
        assert not any(name.endswith(".weight_packed") for name in safetensor_io.index), (
            "Kimi K3 Megatron loading requires the experts to be converted from MXFP4 "
            "to standard BF16 safetensors first"
        )
        return safetensor_io

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name
        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]
        if "self_attention" in mcore_weights_name or "input_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        if "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        return self._weight_name_mapping_other(mcore_weights_name)

    def _weight_to_mcore_format(
        self,
        mcore_weights_name: str,
        hf_weights: list[torch.Tensor],
    ) -> torch.Tensor:
        if mcore_weights_name.endswith(
            (
                "self_attention.q_conv1d.weight",
                "self_attention.k_conv1d.weight",
                "self_attention.v_conv1d.weight",
            )
        ):
            assert len(hf_weights) == 1
            return hf_weights[0].float().contiguous()
        if mcore_weights_name.endswith("self_attention.A_log"):
            assert len(hf_weights) == 1
            return hf_weights[0][: self.config.kimi_linear_num_heads].float().contiguous()
        if mcore_weights_name.endswith("self_attention.dt_bias"):
            assert len(hf_weights) == 1
            return hf_weights[0].float().contiguous()
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
