"""NemotronH (Mamba2 hybrid) adaptations: a post-load packed-doc reset and a clobber-reload that
re-asserts the checkpoint over mixer params transformers' _init_weights re-inits after loading."""

from ..packing.registry import PackingPatch, register_packing_patch
from ..post_load_fixups import PostLoadFixup, register_post_load_fixup


def _is_mamba_hybrid(hf_config) -> bool:
    """True for Mamba/SSM-hybrid archs whose HF `_init_weights` clobbers loaded weights post-load."""
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    if "nemotron_h" in model_type or "mamba" in model_type:
        return True
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return bool(layer_types) and any("mamba" in str(t).lower() for t in layer_types)


def _packing_applies(hf_config) -> bool:
    return "nemotron_h" in str(getattr(hf_config, "model_type", "") or "").lower()


def _packing_apply(model):
    from ...models.nemotron_h import apply_nemotron_h_sglang_match_patch

    return apply_nemotron_h_sglang_match_patch(model)


def _reload_clobbered_from_disk(model, ckpt_path, tol=1e-3) -> int:
    from ...models.nemotron_h import _reload_clobbered_from_disk as reload_impl

    return reload_impl(model, ckpt_path, tol)


register_packing_patch(PackingPatch("nemotron_h_packing", _packing_applies, "post_load", _packing_apply))
register_post_load_fixup(PostLoadFixup("mamba_clobber_reload", _is_mamba_hybrid, _reload_clobbered_from_disk))
