"""HF PEFT names for the torchtitan backend's LoRA adapter params, through the model's own state-dict adapter.

A fused projection (q/k/v, gate/up) carries one ``lora_a`` for all the HF projections it holds and stacks their
``lora_b`` rows exactly like its base weight, so ``lora_b`` splits through the adapter the way the base weight does
and each HF projection gets a copy of the shared ``lora_A``.

A routed-expert projection carries shared-outer adapters: the factor on the model-dimension side (``lora_a`` of
w13, ``lora_b`` of w2) is one matrix for all experts, the other keeps the expert axis. The expert-axis factor
goes through the adapter like the base weight, which names and splits it per expert; the parts are restacked
into one tensor per expert-agnostic module (``experts.gate_up_proj.lora_B`` ``[E, 2F, r]``), and the shared
factor is exported once with an expert axis of 1, the layout the engine serves. Needs only torch, so the weight
iterator imports without torchtitan."""

import re
from collections.abc import Iterator, Mapping

import torch

from miles.utils.lora.hf_lora_targets import collapse_hf_expert_modules

_PEFT_FACTOR_BY_TITAN_NAME = {"lora_a": "lora_A", "lora_b": "lora_B"}
_EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")
# the grouped routed-expert projections of every torchtitan MoE live under this module
ROUTED_EXPERTS = ".routed_experts."


def iter_hf_adapter_tensors(sd_adapter, model_state: Mapping[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """The adapter params in ``model_state`` under HF PEFT names (``...q_proj.lora_A.weight``)."""
    for key, lora_b in model_state.items():
        parts = split_adapter_key(key)
        if parts is None or parts[1] != "lora_b":
            continue
        projection = parts[0]
        lora_a = model_state[f"{projection}.lora_a.weight"]
        if is_grouped_adapter(projection):
            yield from _grouped_hf_adapter_tensors(sd_adapter, projection, lora_a, lora_b)
            continue
        for hf_name, hf_lora_b in sd_adapter.to_hf({f"{projection}.weight": lora_b}).items():
            module = hf_name.removesuffix(".weight")
            yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_a']}.weight", lora_a
            yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_b']}.weight", hf_lora_b


def is_grouped_adapter(projection: str) -> bool:
    """Whether the adapter belongs to a routed-expert projection, whose expert-axis factor is restacked below.
    Tensor ranks cannot tell: a dense fused gate/up ``lora_b`` is ``[2, F, r]`` like a shared-A expert ``w13``'s."""
    return ROUTED_EXPERTS in projection


def _grouped_hf_adapter_tensors(sd_adapter, projection, lora_a, lora_b) -> Iterator[tuple[str, torch.Tensor]]:
    expert_factor, shared_factor = ("lora_b", "lora_a") if lora_a.dim() == 2 else ("lora_a", "lora_b")
    expert_tensor, shared_tensor = (lora_b, lora_a) if expert_factor == "lora_b" else (lora_a, lora_b)
    by_module = _restack_experts(sd_adapter.to_hf({f"{projection}.weight": expert_tensor}))
    for module, tensor in by_module.items():
        yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME[expert_factor]}.weight", tensor
        yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME[shared_factor]}.weight", shared_tensor.unsqueeze(0)


def _restack_experts(hf_tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One ``[E, ...]`` tensor per expert-agnostic module from the adapter's per-expert (or packed) output."""
    parts: dict[str, dict[int, torch.Tensor]] = {}
    packed: dict[str, torch.Tensor] = {}
    for hf_name, tensor in hf_tensors.items():
        module = hf_name.removesuffix(".weight")
        match = _EXPERT_INDEX.search(module)
        if match is None:
            packed[module] = tensor
        else:
            parts.setdefault(_EXPERT_INDEX.sub(".experts.", module), {})[int(match.group(1))] = tensor
    stacked = {
        module: torch.stack([by_expert[index] for index in sorted(by_expert)]) for module, by_expert in parts.items()
    }
    stacked.update(packed)
    folded = {}
    for module in collapse_hf_expert_modules(stacked):
        if module in stacked:
            folded[module] = stacked[module]
        else:
            prefix = module.removesuffix(".gate_up_proj")
            folded[module] = torch.cat([stacked[f"{prefix}.gate_proj"], stacked[f"{prefix}.up_proj"]], dim=1)
    return folded


def is_adapter_key(key: str) -> bool:
    return split_adapter_key(key) is not None


def split_adapter_key(key: str) -> tuple[str, str] | None:
    """``(base projection FQN, factor)`` for a titan adapter param key, else None."""
    for factor in _PEFT_FACTOR_BY_TITAN_NAME:
        suffix = f".{factor}.weight"
        if key.endswith(suffix):
            return key.removesuffix(suffix), factor
    return None


def grouped_hf_export_names(sd_adapter, projection: str, weight_shape: tuple[int, ...]) -> list[str]:
    """The expert-agnostic HF modules a routed-expert projection exports as (``...experts.gate_up_proj``)."""
    probe = torch.empty(weight_shape, device="meta")
    try:
        exported = sd_adapter.to_hf({f"{projection}.weight": probe})
    except ValueError:
        return []
    return collapse_hf_expert_modules(name.removesuffix(".weight") for name in exported)


def hf_export_names(sd_adapter, projection: str, weight_shape: tuple[int, ...]) -> list[str]:
    """The HF modules a titan projection's weight exports as: several for a fused projection, none for one the
    adapter does not map (torchtitan's Muse Spark vision encoder builds its ``conv1`` config as the module
    ``conv1_linear``, so that config FQN is no state-dict key; the Muse Spark adapter rejects an unknown key where
    the others skip it, and either way no HF target can name the projection)."""
    probe = torch.empty(weight_shape, device="meta")
    try:
        exported = sd_adapter.to_hf({f"{projection}.weight": probe})
    except ValueError:
        return []
    return [name.removesuffix(".weight") for name in exported]
