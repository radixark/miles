"""Backend-neutral API for streaming training-side weights as HF-named tensors."""

import dataclasses
import itertools
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from typing import ClassVar

import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import (
    AtomicUpdateGroup,
    assemble_atomic_update_groups,
    pack_units_by_size,
)


@dataclasses.dataclass(frozen=True)
class WeightUpdatePlacement:
    """Which training-side parallel dims the iterator gathers before yielding.

    A gathered dim: every yielded tensor is full along that dim on every rank
    of that dim's group. A non-gathered dim: each rank yields its own shard of
    the param set.
    """

    gather_pp: bool
    # Always gathered today; explicit so a future protocol can relax them.
    gather_tp: bool = True
    gather_ep: bool = True


def resolve_placement(required: WeightUpdatePlacement, forced: WeightUpdatePlacement | None) -> WeightUpdatePlacement:
    """Join of the protocol's required placement and the iterator's forced one:
    a dim is gathered if either side gathers it."""
    if forced is None:
        return required
    return WeightUpdatePlacement(
        gather_pp=required.gather_pp or forced.gather_pp,
        gather_tp=required.gather_tp or forced.gather_tp,
        gather_ep=required.gather_ep or forced.gather_ep,
    )


class HfWeightIteratorBase(ABC):
    """Streams a training model's weights as HF-named tensors.

    Collective: every training rank must drive the iterators in lockstep.
    Yielded tensors are freshly allocated per bucket and stay valid while the
    caller holds a reference.
    """

    # Placement this implementation can produce; None means any.
    forced_placement: ClassVar[WeightUpdatePlacement | None] = None
    # Engine runners a sync covers ("all" or "target"); a backend narrows it when its draft is frozen.
    weight_update_selector: str = "all"

    def __init__(
        self,
        args: Namespace,
        model,
        *,
        placement: WeightUpdatePlacement,
        model_name: str,
        quantization_config: dict | None,
    ) -> None:
        self.args = args
        self.model = model
        self.placement = placement
        self.model_name = model_name
        self.quantization_config = quantization_config

    def iter_hf_weights(
        self,
        weights: Mapping[str, torch.Tensor] | None,
        *,
        include_base: bool = True,
        adapters: Sequence[tuple[str, object]] = (),
        materialize: bool = True,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Model weights as size-bounded buckets of HF-named GPU tensors;
        atomic update groups are never split across buckets.

        ``weights``: backend-native named weights to read; None reads the live
        model parameters. ``adapters``: ``(lora_name, adapter_or_None)`` pairs
        whose tensors join the stream under ``{lora_name}:{hf_key}`` names.
        ``materialize=False`` joins every collective but yields nothing.
        """
        hf_param_units = self._iter_hf_param_units(weights, materialize=materialize) if include_base else iter(())
        for lora_name, adapter in adapters:
            hf_param_units = itertools.chain(
                hf_param_units, self._iter_hf_adapter_units(lora_name, adapter, materialize=materialize)
            )
        atomic_update_groups = self._hf_atomic_update_groups() if include_base and materialize else []
        hf_param_units = assemble_atomic_update_groups(hf_param_units, atomic_update_groups)
        yield from pack_units_by_size(hf_param_units, self.args.update_weight_buffer_size)

    @abstractmethod
    def _iter_hf_param_units(
        self,
        weights: Mapping[str, torch.Tensor] | None,
        *,
        materialize: bool,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Backend hook: one unit per training-side parameter, holding every HF
        tensor it converted into (a weight, or weight + quant scales), honoring
        ``self.placement``. Collectives must run lockstep on every rank;
        ``materialize=False`` joins them but yields nothing."""

    def _hf_atomic_update_groups(self) -> list[AtomicUpdateGroup]:
        """Backend hook: HF-namespace atomic groups for this model. Default none."""
        return []

    @abstractmethod
    def _iter_hf_adapter_units(
        self, lora_name: str, adapter, *, materialize: bool
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Backend hook: this rank's slice of the adapter per ``self.placement``,
        one unit per parameter, names ``{lora_name}:{hf_key}``, rank-trimmed.
        Collectives must run lockstep on every rank; ``materialize=False`` joins
        them but yields nothing."""
