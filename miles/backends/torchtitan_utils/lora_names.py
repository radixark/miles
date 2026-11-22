"""HF PEFT names for the torchtitan backend's LoRA adapter params, through the model's own state-dict adapter.

A fused projection (q/k/v, gate/up) carries one ``lora_a`` for all the HF projections it holds and stacks their
``lora_b`` rows exactly like its base weight, so ``lora_b`` splits through the adapter the way the base weight does
and each HF projection gets a copy of the shared ``lora_A``. A grouped expert projection the checkpoint stores as one
tensor (``experts.w13_weight``) exports as the module ``experts.w13`` with its factors stacked over the experts
(``lora_A`` [E, r, in], ``lora_B`` [E, out, r]). Needs only torch, so the weight iterator imports without
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
            module = _hf_module(hf_name)
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
    return [_hf_module(name) for name in sd_adapter.to_hf({f"{projection}.weight": probe})]


def _hf_module(hf_weight_name: str) -> str:
    """``...q_proj.weight`` -> ``...q_proj``; a stacked expert tensor ``...experts.w13_weight`` -> ``...experts.w13``."""
    if hf_weight_name.endswith(".weight"):
        return hf_weight_name.removesuffix(".weight")
    return hf_weight_name.removesuffix("_weight")
