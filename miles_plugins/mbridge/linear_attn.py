class LinearAttentionBridgeMixin:
    """Keeps the linear-attention weight_layout_version marker out of HF weight loading."""

    def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep=True):
        mapping = super()._weight_name_mapping_mcore_local_to_global(model, consider_ep=consider_ep)
        return {k: v for k, v in mapping.items() if not v.endswith("weight_layout_version")}
