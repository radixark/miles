"""SGLang serving export: zero-filled siblings for fused serving buffers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from miles_plugins.lora.hf_adapter import export_lora_hf_named
from miles_plugins.lora.modules.linear import iter_adapters


def export_lora_sglang_named(model_chunks: Sequence[nn.Module]) -> list[tuple[str, torch.Tensor]]:
    """Export native adapter weights in a form every fused SGLang path accepts.

    ``export_lora_hf_named`` is the exact-target representation.  For every fused
    serving group with at least one exported member, absent members get
    zero-valued A/B pairs sized from the group's declared row widths.
    """

    exact = export_lora_hf_named(model_chunks)
    exported = dict(exact)
    assert len(exported) == len(exact), "native LoRA export produced duplicate HF names across model chunks"

    for adapter in iter_adapters(model_chunks):
        prefix = adapter.hf_prefix
        for export in adapter.exports():
            group = export.fused_group
            if group is None:
                continue
            exemplar_a = exported[f"{export.hf_name}.lora_A.weight"]
            for member, rows_full in group.member_rows.items():
                a_name = f"{prefix}{member}.lora_A.weight"
                b_name = f"{prefix}{member}.lora_B.weight"
                if a_name in exported:
                    continue
                exported[a_name] = torch.zeros_like(exemplar_a)
                exported[b_name] = exemplar_a.new_zeros((rows_full, exemplar_a.shape[0]))

    return list(exported.items())


__all__ = ["export_lora_sglang_named"]
