"""Qwen3.5/3.6/Qwen3-Next (GatedDeltaNet) adaptation: a config-time packed-doc reset that feeds
cu_seqlens to fla chunk/recurrent_gated_delta_rule and seq_idx to causal_conv1d_fn per packed document.
Patches the DecoderLayer/GatedDeltaNet class forwards; kernel logic lives in ``models/qwen3_5.py``."""

from miles.backends.fsdp_utils.models.replay_routers import install_qwen3_router_replay
from ..packing.registry import PackingPatch, register_packing_patch
from ..routing_replay import RoutingReplayAdapter, register_routing_replay_adapter


def _applies(hf_config) -> bool:
    """True for GatedDeltaNet archs (Qwen3.5/3.6, Qwen3-Next): a linear_attention layer type or qwen3_5."""
    if hf_config is None:
        return False
    model_type = str(getattr(hf_config, "model_type", "") or "")
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return (layer_types is not None and "linear_attention" in layer_types) or "qwen3_5" in model_type


def _apply():
    from ...models.qwen3_5 import apply_gateddeltanet_packing_patch

    return apply_gateddeltanet_packing_patch()


register_packing_patch(PackingPatch("gated_deltanet_packing", _applies, "config", _apply))


def _is_qwen3_5_moe(hf_config) -> bool:
    """Qwen3.5 MoE; the text-only checkpoint reports model_type qwen3_5_moe_text."""
    if hf_config is None:
        return False
    model_type = str(getattr(hf_config, "model_type", "") or "")
    return "qwen3_5" in model_type and "moe" in model_type


register_routing_replay_adapter(
    RoutingReplayAdapter(
        name="qwen3_5_moe",
        applies_to=_is_qwen3_5_moe,
        module_cls_name="Qwen3_5MoeTopKRouter",
        install=install_qwen3_router_replay,
    )
)
