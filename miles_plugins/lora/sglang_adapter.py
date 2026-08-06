"""SGLang serving export: fused-buffer zero-fill siblings and target expansion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn

from miles_plugins.lora.hf_adapter import export_lora_hf_named
from miles_plugins.lora.modules.linear import iter_adapters


def expand_sglang_target_modules(target_modules: Iterable[str]) -> list[str]:
    """Expand logical split targets to the fused-buffer projection families.

    SGLang normalizes every Q/K/V target to ``qkv_proj`` and every gate/up
    target to ``gate_up_proj``.  Advertising all logical siblings keeps its
    adapter config consistent with the zero-padded serving export.  The
    families come from the registered layout declarations, so a new
    architecture's fused groups extend this expansion automatically.
    """
    from miles_plugins.lora.registry import serving_fused_families

    targets = list(dict.fromkeys(target_modules))
    target_set = set(targets)
    for family in serving_fused_families():
        if target_set.intersection(family):
            targets.extend(name for name in sorted(family) if name not in target_set)
            target_set.update(family)
    return targets


def sglang_target_modules(args) -> list[str]:
    """The ``lora_target_modules`` SGLang should be launched and synced with.

    A spec that declares ``sglang_lora_target_modules`` hands the engine that
    literal list (Inkling's TML names are engine-detected via ``["all"]``);
    everyone else expands the run's effective targets to SGLang's fused
    families.
    """
    from miles_plugins.lora.config import LoRAConfig
    from miles_plugins.lora.registry import _resolve_registered_spec, resolve_native_lora_config

    if getattr(args, "hf_checkpoint", None):
        _model_type, spec = _resolve_registered_spec(args.hf_checkpoint)
        if spec.sglang_lora_target_modules is not None:
            return list(spec.sglang_lora_target_modules)
        effective_targets = resolve_native_lora_config(args).target_modules
    else:
        effective_targets = LoRAConfig.from_args(args).target_modules
    return expand_sglang_target_modules(sorted(effective_targets))


def export_lora_sglang_named(model_chunks: Sequence[nn.Module]) -> list[tuple[str, torch.Tensor]]:
    """Export native adapter weights in a form every fused SGLang path accepts.

    Only serving sync uses this entry point.  ``export_lora_hf_named`` remains
    the lossless, exact-target checkpoint representation.  For every fused
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


__all__ = ["expand_sglang_target_modules", "export_lora_sglang_named", "sglang_target_modules"]
