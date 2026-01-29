from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("deepseek_v32")
class DeepseekV32Bridge(DeepseekV3Bridge):

    # Weights with parallel_mode="duplicated" that should NOT be gathered across TP
    _DUPLICATED_WEIGHTS = {
        "self_attention.core_attention.indexer.linear_wq_b.weight",
        "self_attention.core_attention.indexer.linear_wk.weight",
        "self_attention.core_attention.indexer.linear_weights_proj.weight",
    }

    _ATTENTION_MAPPING = DeepseekV3Bridge._ATTENTION_MAPPING.copy()

    # Because the indexer needs the norm output, we cannot use the fused transformer engine impl and have to compute it separately.
    if "self_attention.linear_q_up_proj.layer_norm_weight" in _ATTENTION_MAPPING:
        del _ATTENTION_MAPPING["self_attention.linear_q_up_proj.layer_norm_weight"]
    if "self_attention.linear_kv_up_proj.layer_norm_weight" in _ATTENTION_MAPPING:
        del _ATTENTION_MAPPING["self_attention.linear_kv_up_proj.layer_norm_weight"]

    _ATTENTION_MAPPING.update(
        {
            "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_a_layernorm.weight"],
            "self_attention.kv_layernorm.weight": ["model.layers.{layer_number}.self_attn.kv_a_layernorm.weight"],
            "self_attention.core_attention.indexer.linear_wq_b.weight": [
                "model.layers.{layer_number}.self_attn.indexer.wq_b.weight"
            ],
            "self_attention.core_attention.indexer.linear_wk.weight": [
                "model.layers.{layer_number}.self_attn.indexer.wk.weight"
            ],
            "self_attention.core_attention.indexer.k_norm.weight": [
                "model.layers.{layer_number}.self_attn.indexer.k_norm.weight"
            ],
            "self_attention.core_attention.indexer.k_norm.bias": [
                "model.layers.{layer_number}.self_attn.indexer.k_norm.bias"
            ],
            "self_attention.core_attention.indexer.linear_weights_proj.weight": [
                "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
            ],
        }
    )

    def _build_config(self):
        config = super()._build_config()

        config.attention_backend = AttnBackend.auto

        config.experimental_attention_variant = "dsa"
        config.dsa_indexer_n_heads = getattr(self.hf_config, "dsa_indexer_n_heads", 64)
        config.dsa_indexer_head_dim = getattr(self.hf_config, "dsa_indexer_head_dim", 128)
        config.dsa_indexer_topk = getattr(self.hf_config, "dsa_indexer_topk", 2048)

        return config
