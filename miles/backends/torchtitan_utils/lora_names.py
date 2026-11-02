"""HF PEFT names for the torchtitan backend's LoRA adapter params, through the model's own state-dict adapter.

A fused projection (q/k/v, gate/up) carries one ``lora_a`` for all the HF projections it holds and stacks their
``lora_b`` rows exactly like its base weight, so ``lora_b`` splits through the adapter the way the base weight does
and each HF projection gets a copy of the shared ``lora_A``. Needs only torch, so the weight iterator imports without
torchtitan."""

from collections.abc import Iterator, Mapping

import torch

_PEFT_FACTOR_BY_TITAN_NAME = {"lora_a": "lora_A", "lora_b": "lora_B"}


def iter_hf_adapter_tensors(sd_adapter, model_state: Mapping[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """The adapter params in ``model_state`` under HF PEFT names (``...q_proj.lora_A.weight``)."""
    for key, lora_b in model_state.items():
        parts = split_adapter_key(key)
        if parts is None or parts[1] != "lora_b":
            continue
        projection = parts[0]
        lora_a = model_state[f"{projection}.lora_a.weight"]
        for hf_name, hf_lora_b in sd_adapter.to_hf({f"{projection}.weight": lora_b}).items():
            module = hf_name.removesuffix(".weight")
            yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_a']}.weight", lora_a
            yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_b']}.weight", hf_lora_b


def is_adapter_key(key: str) -> bool:
    return split_adapter_key(key) is not None


def split_adapter_key(key: str) -> tuple[str, str] | None:
    """``(base projection FQN, factor)`` for a titan adapter param key, else None."""
    for factor in _PEFT_FACTOR_BY_TITAN_NAME:
        suffix = f".{factor}.weight"
        if key.endswith(suffix):
            return key.removesuffix(suffix), factor
    return None


def hf_export_names(sd_adapter, projection: str, weight_shape: tuple[int, ...]) -> list[str]:
    """The HF modules a titan projection's weight exports as; several for a fused projection."""
    probe = torch.empty(weight_shape, device="meta")
    return [name.removesuffix(".weight") for name in sd_adapter.to_hf({f"{projection}.weight": probe})]
