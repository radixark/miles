import torch

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("deepseek_v32")
class DeepseekV32Bridge(DeepseekV3Bridge):
    _DSA_ATTENTION_MAPPING = {
        "self_attention.wq_b.weight": ["model.layers.{layer_number}.self_attn.indexer.wq_b.weight"],
        "self_attention.wk.weight": ["model.layers.{layer_number}.self_attn.indexer.wk.weight"],
        "self_attention.weights_proj.weight": ["model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"],
        "self_attention.k_norm.weight": ["model.layers.{layer_number}.self_attn.indexer.k_norm.weight"],
        "self_attention.k_norm.bias": ["model.layers.{layer_number}.self_attn.indexer.k_norm.bias"],
    }
    _ATTENTION_MAPPING = {**DeepseekV3Bridge._ATTENTION_MAPPING, **_DSA_ATTENTION_MAPPING}

    def _prepare_rope_fields(self):
        """Normalize rope fields."""
        rope_scaling = getattr(self.hf_config, "rope_scaling", None)
        rope_parameters = getattr(self.hf_config, "rope_parameters", None)

        if isinstance(rope_scaling, dict):
            # V3.2 style
            for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
                value = rope_scaling.get(key)
                if isinstance(value, int):
                    rope_scaling[key] = float(value)

            self.hf_config.rope_scaling = rope_scaling
            rope_theta = getattr(self.hf_config, "rope_theta", None)
            if rope_theta is None:
                rope_theta = rope_scaling.get("rope_theta")
            if rope_theta is None:
                raise ValueError("V3.2 rope_scaling must contain rope_theta")
            self.hf_config.rope_theta = float(rope_theta)
            return
        if isinstance(rope_parameters, dict):
            # GLM-5 style
            self.hf_config.rope_theta = float(rope_parameters["rope_theta"])
            return
        raise ValueError("Unsupported rope config for deepseek_v32 bridge")

    def _build_config(self):
        self._prepare_rope_fields()
        return super()._build_config()

    def _get_gptmodel_args(self) -> dict:
        self._prepare_rope_fields()
        return super()._get_gptmodel_args()

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Apply rope reordering when exporting DSA attention weights to HF format.

        Our training uses last half for rope while DeepSeek uses first half,
        so we swap the two halves.
        """
        if not self.hf_config.indexer_rope_interleave:
            return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

        if "self_attention.wq_b.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            wq_b = mcore_weights
            wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
            wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
            return hf_names, [wq_b]
        elif "self_attention.wk.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            wk = mcore_weights
            wk = torch.cat([wk[64:], wk[:64]], dim=0)
            return hf_names, [wk]
        elif "self_attention.k_norm.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            knorm_weight = mcore_weights
            knorm_weight = torch.cat([knorm_weight[64:], knorm_weight[:64]], dim=0)
            return hf_names, [knorm_weight]
        elif "self_attention.k_norm.bias" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            knorm_bias = mcore_weights
            knorm_bias = torch.cat([knorm_bias[64:], knorm_bias[:64]], dim=0)
            return hf_names, [knorm_bias]
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        """Apply inverse rope reordering when importing DSA attention weights from HF format.

        The swap operation is its own inverse: swap the two halves back.
        """
        if not self.hf_config.indexer_rope_interleave:
            return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

        if "self_attention.wq_b.weight" in mcore_weights_name:
            wq_b = hf_weights[0]
            wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
            wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
            return wq_b
        elif "self_attention.wk.weight" in mcore_weights_name:
            wk = hf_weights[0]
            wk = torch.cat([wk[64:], wk[:64]], dim=0)
            return wk
        elif "self_attention.k_norm.weight" in mcore_weights_name:
            knorm_weight = hf_weights[0]
            knorm_weight = torch.cat([knorm_weight[64:], knorm_weight[:64]], dim=0)
            return knorm_weight
        elif "self_attention.k_norm.bias" in mcore_weights_name:
            knorm_bias = hf_weights[0]
            knorm_bias = torch.cat([knorm_bias[64:], knorm_bias[:64]], dim=0)
            return knorm_bias
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)


@register_model("glm_moe_dsa")
class GlmMoeDsaBridge(DeepseekV32Bridge):
    pass
