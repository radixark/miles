from mbridge.core import register_model
from miles_plugins.mbridge.kimi_k3 import KimiK3Bridge
from miles_plugins.models.kimi_k3.model import build_kimi_k3_spec
from miles_plugins.models.kimi_k31.model import KimiK31DSAAttention, configure_kimi_k31_dsa


@register_model("kimi_k31")
class KimiK31Bridge(KimiK3Bridge):
    _INDEXER_MAPPING = {
        f"self_attention.indexer.{name}": [f"language_model.model.layers.{{layer_number}}.self_attn.indexer.{name}"]
        for name in ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight")
    }
    # the indexer keys go first: the lookup takes the first keyword contained in the name
    _ATTENTION_MAPPING = {**_INDEXER_MAPPING, **KimiK3Bridge._ATTENTION_MAPPING}

    def _build_config(self):
        config = super()._build_config()
        # conversion only needs the parameter layout; training picks the backend from its args
        configure_kimi_k31_dsa(config, self.text_config.dsa_config, topk_backend="torch")
        return config

    def _get_transformer_layer_spec(self, vp_stage=None):
        self.has_vp_stage = True
        return build_kimi_k3_spec(self.config, vp_stage=vp_stage, mla_attention=KimiK31DSAAttention)
