"""Merged-weight view of a native-LoRA model for full-model HF export and merged rollout sync."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters


class MergedWeights(Mapping[str, torch.Tensor]):
    """``weights`` with every adapted host weight read as ``weight + delta``.

    Deltas are computed on access in the host's local layout, so the base-weight
    converters gather and rename merged tensors exactly as they would the base.
    A host weight read from a host-memory copy is merged on the adapter's device.
    """

    def __init__(self, weights: Mapping[str, torch.Tensor], deltas: dict[str, Callable[[], torch.Tensor]]):
        self._weights = weights
        self._deltas = deltas

    @torch.no_grad()
    def __getitem__(self, name: str) -> torch.Tensor:
        weight = self._weights[name]
        delta = self._deltas.get(name)
        if delta is None:
            return weight
        update = delta()
        return (weight.to(update.device, non_blocking=True).float() + update).to(weight.dtype)

    def __iter__(self) -> Iterator[str]:
        return iter(self._weights)

    def __len__(self) -> int:
        return len(self._weights)


def merge_lora_into_weights(
    model_chunks: Sequence[nn.Module],
    weights: Mapping[str, torch.Tensor],
    live_names: Mapping[int, str] | None = None,
) -> MergedWeights:
    """Merge every adapter into its host weight in ``weights``.

    ``live_names`` maps a live host parameter's id to its key in ``weights``. Omit it when
    ``weights`` holds the live parameters; pass it when ``weights`` holds copies, such as the
    offload backup a colocated weight sync reads.
    """
    if live_names is None:
        live_names = {id(weight): name for name, weight in weights.items()}
    deltas: dict[str, Callable[[], torch.Tensor]] = {}
    for adapter in iter_adapters(model_chunks):
        for host_weight, delta in adapter.weight_deltas():
            name = live_names.get(id(host_weight))
            assert name in weights, f"the host weight of adapter {adapter.hf_prefix!r} is not exported"
            assert name not in deltas, f"two adapters merge into one host weight of {adapter.hf_prefix!r}"
            deltas[name] = delta
    return MergedWeights(weights, deltas)
