"""Prepare small derived consumer values without modifying live model state (CUDA receiver only).

Weight and block-scale permutations belong to byte bindings. Kernel backends describe the updates
that recompute complete scalar values from changed scales (``ExpertLayout.expert_derived_updates``);
this module validates their patch geometry once and prepares them, never decoded XOR masks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from itertools import product

import torch

from miles.backends.sglang_utils.gpu_delta.byte_views import _storage_key


@dataclass(frozen=True)
class PreparedCopy:
    destination: torch.Tensor
    source: torch.Tensor

    def refresh(self, ok):
        # The prepared value lands only behind the sticky error gate.
        torch.where(ok, self.source, self.destination, out=self.destination)


@dataclass(frozen=True)
class DerivedUpdate:
    """An update whose inputs can include several independently changed scales.

    ``overrides`` contains complete tensors keyed by ``id(original_source)``.
    The caller assembles any partial canonical updates before preparation and
    applies the returned copies only after successful DE/apply completion.
    ``compute(values)`` takes one tuple with a value per source, the prepared
    complete value where an override exists (``ComputedImage.compute`` takes
    its live sources as separate arguments), and returns one value per
    destination.
    """

    sources: tuple[torch.Tensor, ...]
    destinations: tuple[torch.Tensor, ...]
    compute: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]
    checks: Callable | None = None

    @cached_property
    def compute_device(self):
        return next((source.device for source in self.sources if source.is_cuda), self.sources[0].device)

    @cached_property
    def mixed_sources(self):
        return any(source.device != self.compute_device for source in self.sources)

    def _prepare(self, overrides):
        # Device predicates are returned unchecked: prepare_updates validates
        # every update's predicates once, before returning any prepared copies.
        values = tuple(overrides.get(id(source), source) for source in self.sources)
        with torch.no_grad():
            if self.mixed_sources:
                values = tuple(value.to(self.compute_device) for value in values)
            outputs = tuple(self.compute(values))
            if len(outputs) != len(self.destinations):
                raise ValueError("GPU delta derived output count changed")
            result, snapshots = [], {}
            for destination, value in zip(self.destinations, outputs, strict=False):
                _same_geometry(destination, value)
                # Retain complete target values independently of live buffers;
                # later source copies must not alter a prepared update.
                if id(value) not in snapshots:
                    snapshots[id(value)] = value.detach().clone()
                result.append(PreparedCopy(destination, snapshots[id(value)]))
            checks = self.checks(outputs) if self.checks is not None else ()
        return tuple(result), checks


def _validate_checks(checks):
    if not checks:
        return
    # Batch each device's predicates; CPU-resident calibration is valid too.
    by_device = {}
    for predicate, message in checks:
        by_device.setdefault(predicate.device, []).append((predicate, message))
    for group in by_device.values():
        passed = torch.stack([predicate for predicate, _ in group]).cpu().tolist()
        for ok, (_, message) in zip(passed, group, strict=False):
            if not ok:
                raise ValueError(message)


def _same_geometry(destination, value):
    # A derived result may be computed on CUDA for a CPU consumer (or vice
    # versa). The paused commit owns the cross-device copy.
    if not isinstance(value, torch.Tensor) or destination.shape != value.shape or destination.dtype != value.dtype:
        raise ValueError("GPU delta derived destination geometry changed")


def _element_offsets(tensor):
    return [
        tensor.storage_offset() + sum(i * stride for i, stride in zip(index, tensor.stride(), strict=False))
        for index in product(*(range(size) for size in tensor.shape))
    ]


def plan_updates(updates, targets):
    """Validate and cache scalar patch geometry once at admission.

    Each plan is ``(update, ((source, targets), ...))``.
    """
    by_storage = {}
    for target in targets:
        by_storage.setdefault(_storage_key(target), []).append(target)
    plans = []
    for update in updates:
        sources = []
        for original in update.sources:
            patches = by_storage.get(_storage_key(original), ())
            if not patches:
                continue
            if not original.is_contiguous():
                raise ValueError("GPU delta derived source must be contiguous")
            start, stop = (
                original.storage_offset(),
                original.storage_offset() + original.numel(),
            )
            written = set()
            for target in patches:
                if target.dtype != original.dtype:
                    raise ValueError("GPU delta scalar view dtype mismatch")
                offsets = set(_element_offsets(target))
                if any(offset < start or offset >= stop for offset in offsets):
                    raise ValueError("GPU delta scalar view is outside its source")
                if len(offsets) != target.numel() or offsets & written:
                    raise ValueError("GPU delta scalar updates overlap")
                written.update(offsets)
            sources.append((original, tuple(patches)))
        plans.append((update, tuple(sources)))
    return tuple(plans)


def prepare_updates(plans, raw_values):
    """Patch cached scalar views, then validate device predicates once per update.

    ``raw_values`` maps ``id(target)`` of a raw binding target to its predicted value.
    """
    prepared, checks = [], []
    for update, sources in plans:
        overrides = {}
        for original, targets in sources:
            changed = [target for target in targets if id(target) in raw_values]
            if not changed:
                continue
            # Admission proves the patches are disjoint and within this source.
            # Read live values only when some elements remain unchanged.
            complete = (
                torch.empty_like(original)
                if sum(target.numel() for target in changed) == original.numel()
                else original.detach().clone()
            )
            destinations, values = [], []
            for target in changed:
                offset = target.storage_offset() - original.storage_offset()
                destinations.append(complete.as_strided(target.shape, target.stride(), offset))
                values.append(raw_values[id(target)])
            torch._foreach_copy_(destinations, values)
            overrides[id(original)] = complete
        if overrides:
            copies, predicates = update._prepare(overrides)
            prepared.extend(copies)
            checks.extend(predicates)
    _validate_checks(checks)
    return tuple(prepared)
