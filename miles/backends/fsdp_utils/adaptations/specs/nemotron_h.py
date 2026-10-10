"""NemotronH (Mamba2 hybrid): a config-time repair of sglang's stale pattern-parsing monkeypatch and a
packed-doc reset patch."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter


def _repair_pattern_to_list() -> None:
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


class NemotronHAdapter(ArchAdapter):
    model_types = frozenset({"nemotron_h"})
    verified = True

    def patch_classes(self, args):
        _repair_pattern_to_list()

    def patch_model(self, model, args):
        # HF's NemotronH mixer hardcodes seq_idx=None, so per-document resets need a patch, not kwargs.
        from miles.backends.fsdp_utils.models.nemotron_h import apply_nemotron_h_sglang_match_patch

        apply_nemotron_h_sglang_match_patch(model)
