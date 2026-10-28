"""HF PEFT names for the torchtitan backend's LoRA adapter params, through the model's own
state-dict adapter. Needs only torch, so the weight iterator imports without torchtitan."""

from collections.abc import Iterator, Mapping

import torch

_PEFT_FACTOR_BY_TITAN_NAME = {"lora_a": "lora_A", "lora_b": "lora_B"}


def iter_hf_adapter_tensors(sd_adapter, model_state: Mapping[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """The adapter params in ``model_state`` under HF PEFT names (``...q_proj.lora_A.weight``)."""
    for key, tensor in model_state.items():
        parts = split_adapter_key(key)
        if parts is None:
            continue
        projection, factor = parts
        yield f"{hf_projection_name(sd_adapter, projection)}.{_PEFT_FACTOR_BY_TITAN_NAME[factor]}.weight", tensor


def is_adapter_key(key: str) -> bool:
    return split_adapter_key(key) is not None


def split_adapter_key(key: str) -> tuple[str, str] | None:
    """``(base projection FQN, factor)`` for a titan adapter param key, else None."""
    for factor in _PEFT_FACTOR_BY_TITAN_NAME:
        suffix = f".{factor}.weight"
        if key.endswith(suffix):
            return key.removesuffix(suffix), factor
    return None


def hf_projection_name(sd_adapter, projection: str) -> str:
    """The HF module a titan projection exports as; LoRA needs a pure rename, since
    the adapter factors cannot go through a split, a fuse or a permutation."""
    probe = torch.empty(0)
    hf_state = sd_adapter.to_hf({f"{projection}.weight": probe})
    if len(hf_state) != 1 or next(iter(hf_state.values())) is not probe:
        raise ValueError(
            f"{projection} exports as {sorted(hf_state)} rather than one renamed HF projection; "
            "the torchtitan backend cannot name a LoRA adapter for it"
        )
    return next(iter(hf_state)).removesuffix(".weight")


def hf_export_names(sd_adapter, projection: str) -> list[str]:
    return [name.removesuffix(".weight") for name in sd_adapter.to_hf({f"{projection}.weight": torch.empty(0)})]
