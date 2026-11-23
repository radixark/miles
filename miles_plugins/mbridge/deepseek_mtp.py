class DeepseekMtpBridgeMixin:
    """Maps the MTP layer onto the HF layer after the last decoder layer, where DeepSeek-layout checkpoints store it."""

    def _convert_mtp_param(self, name: str) -> list[str]:
        assert self.config.mtp_num_layers == 1, "only support one mtp layer for now"
        mtp_layer_id = self.config.num_layers
        direct_name_mapping = {
            "mtp.layers.0.enorm.weight": f"model.layers.{mtp_layer_id}.enorm.weight",
            "mtp.layers.0.hnorm.weight": f"model.layers.{mtp_layer_id}.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": f"model.layers.{mtp_layer_id}.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight": f"model.layers.{mtp_layer_id}.shared_head.norm.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]]
        inner = next((p for p in ("transformer_layer", "mtp_model_layer") if f"mtp.layers.0.{p}" in name), None)
        assert inner is not None, f"mtp not found in {name}"
        proxy_name = name.replace(f"mtp.layers.0.{inner}", f"decoder.layers.{mtp_layer_id}")
        if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
            return self._weight_name_mapping_attention(proxy_name)
        if "mlp" in proxy_name:
            return self._weight_name_mapping_mlp(proxy_name)
        raise NotImplementedError(f"Unsupported parameter name: {name}")
