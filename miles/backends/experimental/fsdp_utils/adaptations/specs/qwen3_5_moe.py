"""Qwen3.5/3.6/Qwen3-Next (GatedDeltaNet) adaptation: config-time packed-doc reset; kernel logic in models/qwen3_5_moe.py."""

from ..packing.registry import PackingPatch, register_packing_patch


def _applies(hf_config) -> bool:
    """True for GatedDeltaNet archs (Qwen3.5/3.6, Qwen3-Next): a linear_attention layer type or qwen3_5."""
    if hf_config is None:
        return False
    model_type = str(getattr(hf_config, "model_type", "") or "")
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return (layer_types is not None and "linear_attention" in layer_types) or "qwen3_5" in model_type


def _apply():
    from ...models.qwen3_5_moe import apply_gateddeltanet_packing_patch

    return apply_gateddeltanet_packing_patch()


register_packing_patch(PackingPatch("gated_deltanet_packing", _applies, "config", _apply))
