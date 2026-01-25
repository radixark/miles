from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("deepseek_v4")
class DeepseekV4Bridge(DeepseekV3Bridge):

    # Weights with parallel_mode="duplicated" that should NOT be gathered across TP
    _DUPLICATED_WEIGHTS = {
        # DSA Indexer weights (V4 has indexer directly under self_attention, not core_attention)
        "self_attention.indexer.linear_wq_b.weight",
        "self_attention.indexer.linear_wk.weight",
        "self_attention.indexer.linear_weights_proj.weight",
        # HC weights are float32 and not sharded
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
    }

    _ATTENTION_MAPPING = DeepseekV3Bridge._ATTENTION_MAPPING.copy()

    # Remove fused layernorm mappings (V4 uses separate layernorms)
    _ATTENTION_MAPPING.pop("self_attention.linear_q_up_proj.layer_norm_weight", None)
    _ATTENTION_MAPPING.pop("self_attention.linear_kv_up_proj.layer_norm_weight", None)

    # V4 Attention uses different naming (wq_a, wkv, etc.)
    _ATTENTION_MAPPING.update(
        {
            # Q projection
            "self_attention.wq_a.weight": ["model.layers.{layer_number}.self_attn.wq_a.weight"],
            "self_attention.q_norm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
            "self_attention.wq_b.weight": ["model.layers.{layer_number}.self_attn.wq_b.weight"],
            # KV projection
            "self_attention.wkv.weight": ["model.layers.{layer_number}.self_attn.wkv.weight"],
            "self_attention.kv_norm.weight": ["model.layers.{layer_number}.self_attn.kv_norm.weight"],
            # O projection
            "self_attention.wo_a.weight": ["model.layers.{layer_number}.self_attn.wo_a.weight"],
            "self_attention.wo_b.weight": ["model.layers.{layer_number}.self_attn.wo_b.weight"],
            # Attention sink
            "self_attention.attn_sink": ["model.layers.{layer_number}.self_attn.attn_sink"],
            # Compressor
            "self_attention.compressor.ape": ["model.layers.{layer_number}.self_attn.compressor.ape"],
            "self_attention.compressor.wkv.weight": ["model.layers.{layer_number}.self_attn.compressor.wkv.weight"],
            "self_attention.compressor.wgate.weight": ["model.layers.{layer_number}.self_attn.compressor.wgate.weight"],
            "self_attention.compressor.norm.weight": ["model.layers.{layer_number}.self_attn.compressor.norm.weight"],
            # DSA Indexer (V4 has indexer directly under self_attention)
            "self_attention.indexer.linear_wq_b.weight": [
                "model.layers.{layer_number}.self_attn.indexer.wq_b.weight"
            ],
            "self_attention.indexer.linear_weights_proj.weight": [
                "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
            ],
            # Indexer's nested compressor
            "self_attention.indexer.compressor.ape": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.ape"
            ],
            "self_attention.indexer.compressor.wkv.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wkv.weight"
            ],
            "self_attention.indexer.compressor.wgate.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wgate.weight"
            ],
            "self_attention.indexer.compressor.norm.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.norm.weight"
            ],
            # Hyper-Connection (layer level, directly under layer, not self_attn)
            "hc_attn_fn": ["model.layers.{layer_number}.hc_attn_fn"],
            "hc_attn_base": ["model.layers.{layer_number}.hc_attn_base"],
            "hc_attn_scale": ["model.layers.{layer_number}.hc_attn_scale"],
            "hc_ffn_fn": ["model.layers.{layer_number}.hc_ffn_fn"],
            "hc_ffn_base": ["model.layers.{layer_number}.hc_ffn_base"],
            "hc_ffn_scale": ["model.layers.{layer_number}.hc_ffn_scale"],
        }
    )

    _MLP_MAPPING = DeepseekV3Bridge._MLP_MAPPING.copy()
    _MLP_MAPPING.update(
        {
            # Hash routing tid2eid
            "mlp.router.tid2eid": ["model.layers.{layer_number}.mlp.topk.tid2eid"],
        }
    )

    _DIRECT_MAPPING = DeepseekV3Bridge._DIRECT_MAPPING.copy()
    _DIRECT_MAPPING.update(
        {
            "decoder.hc_head_fn": "model.hc_head_fn",
            "decoder.hc_head_base": "model.hc_head_base",
            "decoder.hc_head_scale": "model.hc_head_scale",
        }
    )

    def _build_config(self):
        config = super()._build_config()

        config.attention_backend = AttnBackend.auto

        # DSA Indexer config (same as V3.2)
        config.experimental_attention_variant = "dsv4"
        config.dsa_indexer_n_heads = getattr(self.hf_config, "index_n_heads", 64)
        config.dsa_indexer_head_dim = getattr(self.hf_config, "index_head_dim", 128)
        config.dsa_indexer_topk = getattr(self.hf_config, "index_topk", 512)

        # V4 Hyper-Connection config
        config.dsv4_hc_mult = getattr(self.hf_config, "hc_mult", 4)
        config.dsv4_hc_sinkhorn_iters = getattr(self.hf_config, "hc_sinkhorn_iters", 20)
        config.dsv4_hc_eps = getattr(self.hf_config, "hc_eps", 1e-6)

        # V4 Compress config
        config.dsv4_compress_ratios = getattr(self.hf_config, "compress_ratios", None)
        config.dsv4_compress_rope_theta = getattr(self.hf_config, "compress_rope_theta", 160000)

        # V4 Attention config
        config.dsv4_o_groups = getattr(self.hf_config, "o_groups", 8)
        config.dsv4_o_lora_rank = getattr(self.hf_config, "o_lora_rank", 1024)
        config.dsv4_n_hash_layers = getattr(self.hf_config, "n_hash_layers", 3)
        config.dsv4_window_size = getattr(self.hf_config, "window_size", 128)

        return config
