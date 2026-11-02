"""PEFT and official Tinker tensor names <-> HF keys (``<hf module>.lora_{A,B}``); pure functions."""

import re
from dataclasses import dataclass

from miles.utils.lora.hf_lora_targets import HfLoraTargets

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
    # HF PEFT prefixes the model path; miles' exports (main's and this library's) use the bare HF path
    assert name.endswith(".weight"), f"not a PEFT LoRA tensor name: {name!r}"
    module, factor = name.removeprefix(_PEFT_PREFIX).removesuffix(".weight").rsplit(".", 1)
    assert factor in ("lora_A", "lora_B"), f"not a LoRA factor: {name!r}"
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


def map_sampler_keys_to_hf_keys(name: str) -> HfKey:
    for hf_fragment, sglang_fragment in SAMPLER_EXPERT_NAMES_BY_HF.items():
        name = name.replace(sglang_fragment, hf_fragment)
    return map_peft_keys_to_hf_keys(name)


def map_hf_keys_to_peft_keys(key: str, expert_index: int | None = None) -> str:
    module, factor = key.rsplit(".", 1)
    if expert_index is not None:
        head, tail = module.split(".experts.", 1)
        module = f"{head}.experts.{expert_index}.{tail}"
    return f"{module}.{factor}.weight"


def _hf_layer_prefix(targets: HfLoraTargets) -> str:
    any_target = (targets.attention + targets.mlp)[0]
    return any_target.split(".*.", 1)[0] + "."


def map_tinker_keys_to_hf_keys(name: str, targets: HfLoraTargets) -> HfKey:
    """Official Tinker export name -> HF key; experts stay stacked 3D."""
    assert targets.tinker_name_renames, "no official Tinker export of this model family has been checked"
    module, factor = _split_weight_name(name)
    if module == _TINKER_UNEMBED:
        return HfKey(f"{targets.unembed[0]}.{factor}")
    assert module.startswith(_TINKER_LAYER_PREFIX), f"unexpected Tinker tensor name: {name!r}"
    module = _hf_layer_prefix(targets) + module[len(_TINKER_LAYER_PREFIX) :]
    for tinker_fragment, hf_fragment in targets.tinker_name_renames:
        module = module.replace(tinker_fragment, hf_fragment)
    return HfKey(f"{module}.{factor}")


def map_hf_keys_to_tinker_keys(key: str, targets: HfLoraTargets) -> str:
    module, factor = key.rsplit(".", 1)
    if module == targets.unembed[0]:
        return f"{_PEFT_PREFIX}{_TINKER_UNEMBED}.{factor}.weight"
    hf_layer_prefix = _hf_layer_prefix(targets)
    assert module.startswith(hf_layer_prefix), f"unexpected HF key: {key!r}"
    module = _TINKER_LAYER_PREFIX + module[len(hf_layer_prefix) :]
    for tinker_fragment, hf_fragment in targets.tinker_name_renames:
        module = module.replace(hf_fragment, tinker_fragment)
    return f"{_PEFT_PREFIX}{module}.{factor}.weight"


def is_tinker_only_name(name: str, targets: HfLoraTargets) -> bool:
    """True when ``name`` exists only in the official Tinker naming (head, stacked experts or layer prefix)."""
    module, _ = _split_weight_name(name)
    if module == _TINKER_UNEMBED or any(t in module for t, _ in targets.tinker_name_renames):
        return True
    # Tinker drops the VL wrapper level (model.language_model.layers.* -> model.layers.*)
    return module.startswith(_TINKER_LAYER_PREFIX) and _hf_layer_prefix(targets) != _TINKER_LAYER_PREFIX
