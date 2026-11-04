"""PEFT and official Tinker tensor names <-> HF keys (``<hf module>.lora_{A,B}``); pure functions."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError
from miles.utils.lora.hf_lora_targets import GDN_IN_PROJ_COMPONENTS_BY_MODULE, HfLoraTargets

_PEFT_PREFIX = "base_model.model."
_TINKER_LAYER_PREFIX = "model.layers."
_TINKER_UNEMBED = "model.unembed_tokens"
_PER_EXPERT = re.compile(r"^(?P<head>.*\.experts)\.(?P<index>\d+)\.(?P<tail>.*)$")


@dataclass(frozen=True)
class HfKey:
    """``key`` is ``<hf module>.lora_A`` or ``.lora_B``; ``expert_index`` is set for a per-expert 2D PEFT tensor."""

    key: str
    expert_index: int | None = None


def _split_weight_name(name: str) -> tuple[str, str]:
    # HF PEFT prefixes the model path; main's exports and sampler exports use the bare HF path
    if not name.endswith(".weight"):
        raise AdapterSourceError(f"not a PEFT LoRA tensor name: {name!r}")
    module, factor = name.removeprefix(_PEFT_PREFIX).removesuffix(".weight").rsplit(".", 1)
    if factor not in ("lora_A", "lora_B"):
        raise AdapterSourceError(f"not a LoRA factor: {name!r}")
    return module, factor


def map_peft_keys_to_hf_keys(name: str) -> HfKey:
    module, factor = _split_weight_name(name)
    match = _PER_EXPERT.match(module)
    if match is None:
        return HfKey(f"{module}.{factor}")
    return HfKey(f"{match['head']}.{match['tail']}.{factor}", int(match["index"]))


# sglang's stacked-expert names, which sampler exports use (main's native export serves the same)
SAMPLER_EXPERT_NAMES_BY_HF = {
    ".experts.gate_proj.": ".experts.w1.",
    ".experts.up_proj.": ".experts.w3.",
    ".experts.down_proj.": ".experts.w2.",
}


def rename_sampler_experts_to_hf(name: str, targets: HfLoraTargets) -> str:
    """Inverse of the sampler export's renaming; families whose HF experts are already w1 / w2 / w3 keep them."""
    if not any(target.rsplit(".", 1)[1] in ("w1", "w2", "w3") for target in targets.mlp):
        for hf_fragment, sglang_fragment in SAMPLER_EXPERT_NAMES_BY_HF.items():
            name = name.replace(sglang_fragment, hf_fragment)
    return name


def map_sampler_keys_to_hf_keys(name: str, targets: HfLoraTargets) -> HfKey:
    return map_peft_keys_to_hf_keys(rename_sampler_experts_to_hf(name, targets))


def map_hf_keys_to_peft_keys(key: str, expert_index: int | None = None) -> str:
    module, factor = key.rsplit(".", 1)
    if expert_index is not None:
        head, tail = module.split(".experts.", 1)
        module = f"{head}.experts.{expert_index}.{tail}"
    return f"{module}.{factor}.weight"


def map_hf_keys_to_peft_export_keys(key: str, expert_index: int | None = None) -> str:
    # HF PEFT's set_peft_model_state_dict matches only names under the PeftModel's base_model.model.
    return _PEFT_PREFIX + map_hf_keys_to_peft_keys(key, expert_index)


def _hf_layer_prefix(targets: HfLoraTargets) -> str:
    any_target = (targets.attention + targets.mlp)[0]
    return any_target.split(".*.", 1)[0] + "."


def map_tinker_keys_to_hf_keys(name: str, targets: HfLoraTargets) -> HfKey:
    """Official Tinker export name -> HF key; experts stay stacked 3D."""
    assert targets.tinker_name_renames, "no official Tinker export of this model family has been checked"
    module, factor = _split_weight_name(name)
    if module == _TINKER_UNEMBED:
        return HfKey(f"{targets.unembed[0]}.{factor}")
    if not module.startswith(_TINKER_LAYER_PREFIX):
        raise AdapterSourceError(f"unexpected Tinker tensor name: {name!r}")
    module = _hf_layer_prefix(targets) + module[len(_TINKER_LAYER_PREFIX) :]
    for tinker_fragment, hf_fragment in targets.tinker_name_renames:
        module = module.replace(tinker_fragment, hf_fragment)
    return HfKey(f"{module}.{factor}")


def map_hf_keys_to_tinker_keys(key: str, targets: HfLoraTargets) -> str:
    assert targets.tinker_name_renames, "no official Tinker export of this model family has been checked"
    module, factor = key.rsplit(".", 1)
    if module == targets.unembed[0]:
        return f"{_PEFT_PREFIX}{_TINKER_UNEMBED}.{factor}.weight"
    hf_layer_prefix = _hf_layer_prefix(targets)
    assert module.startswith(hf_layer_prefix), f"unexpected HF key: {key!r}"
    module = _TINKER_LAYER_PREFIX + module[len(hf_layer_prefix) :]
    for tinker_fragment, hf_fragment in targets.tinker_name_renames:
        module = module.replace(hf_fragment, tinker_fragment)
    return f"{_PEFT_PREFIX}{module}.{factor}.weight"


def is_hf_only_name(name: str, targets: HfLoraTargets) -> bool:
    """True when ``name`` exists only in HF naming (per-expert index, HF head, or a VL layer prefix Tinker drops)."""
    module, _ = _split_weight_name(name)
    hf_layer_prefix = _hf_layer_prefix(targets)
    return (
        _PER_EXPERT.match(module) is not None
        or module == targets.unembed[0]
        or (hf_layer_prefix != _TINKER_LAYER_PREFIX and module.startswith(hf_layer_prefix))
    )


def is_tinker_only_name(name: str, targets: HfLoraTargets) -> bool:
    """True when ``name`` exists only in the official Tinker naming (head, stacked experts or layer prefix)."""
    module, _ = _split_weight_name(name)
    if module == _TINKER_UNEMBED or any(t in module for t, _ in targets.tinker_name_renames):
        return True
    # Tinker drops the VL wrapper level (model.language_model.layers.* -> model.layers.*)
    return module.startswith(_TINKER_LAYER_PREFIX) and _hf_layer_prefix(targets) != _TINKER_LAYER_PREFIX


_PACKED_GATE_UP = "gate_up_proj"
_GATE_UP_COMPONENTS = ("gate_proj", "up_proj")


def fused_leaves(parent: str, targets: HfLoraTargets) -> dict[str, tuple[str, ...]]:
    """HF-fused module leaf -> the component leaves it splits into, for modules under ``parent``."""
    leaves = {leaf: components for leaf, components in GDN_IN_PROJ_COMPONENTS_BY_MODULE.items() if len(components) > 1}
    if targets.packed_expert_gate_up_layout is not None and parent.endswith(".experts"):
        leaves[_PACKED_GATE_UP] = _GATE_UP_COMPONENTS
    return leaves


def fused_components(module: str, targets: HfLoraTargets) -> tuple[str, ...] | None:
    """The components an HF-fused module's adapter splits into; they share its A. None for other modules."""
    parent, _, leaf = module.rpartition(".")
    components = fused_leaves(parent, targets).get(leaf) if parent else None
    return None if components is None else tuple(f"{parent}.{c}" for c in components)


def split_fused_lora_b(module: str, value: torch.Tensor, targets: HfLoraTargets) -> list[torch.Tensor]:
    """An HF-fused module's B (output rows on dim -2) as its components' Bs."""
    leaf = module.rpartition(".")[2]
    if leaf == _PACKED_GATE_UP:
        assert targets.packed_expert_gate_up_layout == "interleaved", targets.packed_expert_gate_up_layout
        return [value[..., 0::2, :], value[..., 1::2, :]]
    return list(value.split(dict(targets.component_rows_by_fused_module)[leaf], dim=-2))


def fuse_lora_b(module: str, parts: list[torch.Tensor], targets: HfLoraTargets) -> torch.Tensor:
    """Inverse of ``split_fused_lora_b``."""
    if module.rpartition(".")[2] == _PACKED_GATE_UP:
        return torch.stack(parts, dim=-2).flatten(-3, -2)
    return torch.cat(parts, dim=-2)


def map_named_tensors_to_hf_keys(
    named_tensors: Iterable[tuple[str, torch.Tensor]], map_name: Callable[[str], HfKey], targets: HfLoraTargets
) -> tuple[dict[str, torch.Tensor], list[tuple[str, ...]]]:
    """Whole named tensors -> HF-keyed tensors and the HF-fused groups found.

    Per-expert 2D tensors stack to 3D; an HF-fused module (GDN in_proj_qkv, packed experts.gate_up_proj) splits
    into components that share its A.
    """
    per_expert: dict[str, dict[int, torch.Tensor]] = {}
    weights, fused_groups = {}, []
    for name, value in named_tensors:
        hf_key = map_name(name)
        if hf_key.expert_index is not None:
            per_expert.setdefault(hf_key.key, {})[hf_key.expert_index] = value
            continue
        module, factor = hf_key.key.rsplit(".", 1)
        components = fused_components(module, targets)
        if components is None:
            weights[hf_key.key] = value
        elif factor == "lora_A":
            # under the sorted first component, where a shared-A group keeps its A
            weights[f"{min(components)}.lora_A"] = value
            fused_groups.append(components)
        else:
            for component, part in zip(components, split_fused_lora_b(module, value, targets), strict=True):
                weights[f"{component}.lora_B"] = part.clone()
    for key, experts in per_expert.items():
        if sorted(experts) != list(range(len(experts))):
            raise AdapterSourceError(f"{key}: expert indices are not 0..E-1")
        weights[key] = torch.stack([experts[i] for i in range(len(experts))])
    return weights, fused_groups
