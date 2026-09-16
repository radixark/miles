"""NemotronH (Mamba2 hybrid) adaptations: a config-time repair of sglang's stale pattern-parsing
monkeypatch, a post-load packed-doc reset, and a clobber-reload that re-asserts the checkpoint over
mixer params transformers' _init_weights re-inits after loading."""

from ..class_patches import ModelPatchHook, register_model_patch
from ..packing.registry import PackingPatch, register_packing_patch
from ..post_load_fixups import PostLoadFixup, register_post_load_fixup


def _is_nemotron_h(hf_config) -> bool:
    return str(getattr(hf_config, "model_type", "") or "") == "nemotron_h"


def _repair_pattern_to_list(hf_config, args) -> None:
    """``import sglang`` monkeypatches ``NemotronHConfig._pattern_to_list`` to drop unmapped chars,
    silently deleting every ``-`` (MLP) layer, so any transformers-side NemotronH constructed afterwards
    is mis-shaped (wrong depth, shifted attention layers). Probe for that and re-assert the full native
    mapping; a healthy implementation (keeps the MLP entry) is left untouched."""
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    try:
        probe = NemotronHConfig._pattern_to_list("M-")
    except KeyError:
        return  # pre-'-' transformers: no silent corruption to repair; construction will fail loudly
    if probe != ["mamba"]:
        return

    @staticmethod
    def _pattern_to_list(pattern: str) -> list:
        pattern_mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        return [pattern_mapping[char] for char in pattern]

    NemotronHConfig._pattern_to_list = _pattern_to_list


def _packing_applies(hf_config) -> bool:
    return "nemotron_h" in str(getattr(hf_config, "model_type", "") or "").lower()


def _packing_apply(model):
    from ...models.nemotron_h import apply_nemotron_h_sglang_match_patch

    return apply_nemotron_h_sglang_match_patch(model)


def _post_load_apply(model, ckpt_path):
    from ...models.nemotron_h import reload_nemotron_h_clobbered_weights

    return reload_nemotron_h_clobbered_weights(model, ckpt_path)


register_model_patch(ModelPatchHook("nemotron_h_pattern_repair", _is_nemotron_h, _repair_pattern_to_list))
register_packing_patch(PackingPatch("nemotron_h_packing", _packing_applies, "post_load", _packing_apply))
register_post_load_fixup(PostLoadFixup("nemotron_h_clobber_reload", _is_nemotron_h, _post_load_apply))
