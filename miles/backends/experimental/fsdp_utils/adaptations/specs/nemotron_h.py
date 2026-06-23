"""NemotronH (Mamba2 hybrid) adaptations.

Two post-load hooks: the packed-doc reset (resets Mamba2 conv+scan via seq_idx + attention via varlen
cu_seqlens per packed document) and the clobber-reload (re-asserts the checkpoint over mixer params
that transformers' _init_weights re-initializes after loading). Both need the instantiated model.
"""

from ..packing.registry import PackingPatch, register_packing_patch
from ..post_load_fixups import PostLoadFixup, _is_mamba_hybrid, _reload_clobbered_from_disk, register_post_load_fixup


def _packing_applies(hf_config) -> bool:
    return "nemotron_h" in str(getattr(hf_config, "model_type", "") or "").lower()


def _packing_apply(model):
    from ...models.nemotron_h import apply_nemotron_h_sglang_match_patch

    return apply_nemotron_h_sglang_match_patch(model)


register_packing_patch(PackingPatch("nemotron_h_packing", _packing_applies, "post_load", _packing_apply))
register_post_load_fixup(PostLoadFixup("mamba_clobber_reload", _is_mamba_hybrid, _reload_clobbered_from_disk))
