# slime_plugins/models/minimax_bridge.py

"""MiniMax-M2 Bridge implementation."""

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


@register_model("minimax_m2")
class MinimaxM2Bridge(Qwen2MoEBridge):
    
    _ATTENTION_MAPPING = {
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
    }
    
    _MLP_MAPPING = {
        **(Qwen2MoEBridge._MLP_MAPPING),
        "mlp.router.expert_bias": [
            "model.layers.{layer_number}.block_sparse_moe.e_score_correction_bias"
        ],
        "mlp.router.weight": [
            "model.layers.{layer_number}.block_sparse_moe.gate.weight"
        ],
        "mlp.experts.linear_fc1": [
            "model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight",
            "model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight",
        ],
        
        "mlp.experts.linear_fc2": [
            "model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight"
        ],
    }
    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        if "block_sparse_moe" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _build_config(self):
        config= self._build_base_config(
            use_cpu_initialization=False,
            moe_ffn_hidden_size=self.hf_config.intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_local_experts,
            # moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            # moe_router_load_balancing_type="aux_loss",
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="sigmoid",
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            moe_router_pre_softmax=False,
            qk_layernorm=True,
        )
        print(f"moe_router_enable_expert_bias = {config.moe_router_enable_expert_bias}")
        return config
