"""Merged-weight view of a native-LoRA model for full-model HF export."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import iter_adapters


class MergedWeights(Mapping[str, torch.Tensor]):
    """``weights`` with every adapted host weight read as ``weight + delta``.

    Deltas are computed on access in the host's local layout, so the base-weight
    converters gather and rename merged tensors exactly as they would the base.
    """

    def __init__(self, weights: Mapping[str, torch.Tensor], deltas: dict[int, Callable[[], torch.Tensor]]):
        self._weights = weights
        self._deltas = deltas

    @torch.no_grad()
    def __getitem__(self, name: str) -> torch.Tensor:
        weight = self._weights[name]
        delta = self._deltas.get(id(weight))
        if delta is None:
            return weight
        return (weight.float() + delta()).to(weight.dtype)

    def __iter__(self) -> Iterator[str]:
        return iter(self._weights)

    def __len__(self) -> int:
        return len(self._weights)


def merge_lora_into_weights(model_chunks: Sequence[nn.Module], weights: Mapping[str, torch.Tensor]) -> MergedWeights:
    exported = {id(weight) for weight in weights.values()}
    deltas: dict[int, Callable[[], torch.Tensor]] = {}
    for adapter in iter_adapters(model_chunks):
        for host_weight, delta in adapter.weight_deltas():
            assert id(host_weight) in exported, f"the host weight of adapter {adapter.hf_prefix!r} is not exported"
            assert id(host_weight) not in deltas, f"two adapters merge into one host weight of {adapter.hf_prefix!r}"
            deltas[id(host_weight)] = delta
    return MergedWeights(weights, deltas)
