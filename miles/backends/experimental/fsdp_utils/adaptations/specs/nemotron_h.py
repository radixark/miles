"""NemotronH (Mamba2 hybrid) adaptations: a post-load packed-doc reset and a clobber-reload that
re-asserts the checkpoint over mixer params transformers' _init_weights re-inits after loading."""

from ..packing.registry import PackingPatch, register_packing_patch
from ..precision import register_fp32_master_type
from ..post_load_fixups import PostLoadFixup, _is_mamba_hybrid, _reload_clobbered_from_disk, register_post_load_fixup


def _packing_applies(hf_config) -> bool:
    return "nemotron_h" in str(getattr(hf_config, "model_type", "") or "").lower()


def _packing_apply(model):
    from ...models.nemotron_h import apply_nemotron_h_sglang_match_patch

    return apply_nemotron_h_sglang_match_patch(model)


register_packing_patch(PackingPatch("nemotron_h_packing", _packing_applies, "post_load", _packing_apply))
register_post_load_fixup(PostLoadFixup("mamba_clobber_reload", _is_mamba_hybrid, _reload_clobbered_from_disk))
# mixed-dtype checkpoint (mamba A_log/D/dt_bias are fp32) -> uniform fp32 master so FSDP2 wrapping accepts it
register_fp32_master_type("nemotron_h")
