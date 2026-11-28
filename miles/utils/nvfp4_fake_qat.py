"""Megatron-facing adapter for fused NVFP4 fake QAT."""

from __future__ import annotations

import contextvars
import importlib
import os
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

NVFP4_FAKE_QAT_FLAG = "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"


@dataclass(slots=True)
class _CachedAmaxEntry:
    weight_ref: weakref.ReferenceType[torch.Tensor]
    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int
    amax: torch.Tensor


@dataclass(slots=True)
class _AmaxCacheState:
    enabled: bool | None = None
    qdq_config: Any = None
    entries: dict[int, _CachedAmaxEntry] = field(default_factory=dict)


_AMAX_CACHE_VAR: contextvars.ContextVar[_AmaxCacheState | None] = contextvars.ContextVar(
    "miles_nvfp4_fake_qat_amax_cache",
    default=None,
)


@contextmanager
def nvfp4_fake_qat_amax_cache() -> Iterator[None]:
    """Reuse per-expert FP32 weight amaxes across immutable-weight forwards.

    Weights do not change across microbatches or activation-recomputation
    forwards inside a single forward-only or forward-backward pass. Caching the
    4-byte scalar FP32 ``amax`` per expert weight avoids re-reading every expert
    matrix from HBM just to recompute its infinity norm, while keeping full
    non-mutating STE semantics.

    Cache entries hold only weak references to weight tensors and verify
    ``(identity, data_ptr, shape, dtype, device, _version)`` on lookup so
    in-place mutations or address reuse always invalidate safely. All entries
    are cleared unconditionally on exit before any optimizer step or weight
    switch. Nested invocations share the enclosing pass cache.
    """
    if _AMAX_CACHE_VAR.get() is not None:
        yield
        return

    state = _AmaxCacheState()
    token = _AMAX_CACHE_VAR.set(state)
    try:
        yield
    finally:
        state.entries.clear()
        _AMAX_CACHE_VAR.reset(token)


def maybe_fake_quantize_nvfp4_weight_tensors(
    weight_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Apply env-gated fused NVFP4 fake QAT to discrete TE grouped-linear weights."""
    state = _AMAX_CACHE_VAR.get()
    if state is not None and state.enabled is not None:
        if not state.enabled:
            return weight_tensors
    else:
        enabled = os.getenv(NVFP4_FAKE_QAT_FLAG, "0") == "1"
        if state is not None:
            state.enabled = enabled
        if not enabled:
            return weight_tensors

    if not weight_tensors:
        return weight_tensors

    # Keep CuTe DSL optional for every process that does not enable this path.
    qdq_mod = importlib.import_module("miles.kernels.quant.nvfp4_qdq")

    if state is not None and state.qdq_config is not None:
        qdq_config = state.qdq_config
    else:
        qdq_config = qdq_mod.current_nvfp4_qdq_config()
        if state is not None:
            state.qdq_config = qdq_config

    fake_qdq = qdq_mod.fake_nvfp4_quantization_ste
    try:
        compute_amaxes = qdq_mod.compute_nvfp4_amaxes
    except AttributeError:
        return [fake_qdq(weight, qdq_config) for weight in weight_tensors]

    if state is None:
        amaxes = compute_amaxes(weight_tensors)
        return [fake_qdq(weight, qdq_config, amax=amax) for weight, amax in zip(weight_tensors, amaxes, strict=True)]

    amaxes: list[torch.Tensor | None] = [None] * len(weight_tensors)
    missing_indices: list[int] = []
    missing_weights: list[torch.Tensor] = []
    entries = state.entries
    for idx, weight in enumerate(weight_tensors):
        entry = entries.get(id(weight))
        if (
            entry is not None
            and entry.weight_ref() is weight
            and entry.data_ptr == weight.data_ptr()
            and entry.shape == weight.shape
            and entry.dtype == weight.dtype
            and entry.device == weight.device
            and entry.version == weight._version
        ):
            amaxes[idx] = entry.amax
        else:
            missing_indices.append(idx)
            missing_weights.append(weight)

    if missing_weights:
        computed = compute_amaxes(missing_weights)
        for idx, weight, amax in zip(missing_indices, missing_weights, computed, strict=True):
            amaxes[idx] = amax
            entries[id(weight)] = _CachedAmaxEntry(
                weight_ref=weakref.ref(weight),
                data_ptr=weight.data_ptr(),
                shape=tuple(weight.shape),
                dtype=weight.dtype,
                device=weight.device,
                version=weight._version,
                amax=amax,
            )

    return [fake_qdq(weight, qdq_config, amax=amax) for weight, amax in zip(weight_tensors, amaxes, strict=True)]


__all__ = [
    "NVFP4_FAKE_QAT_FLAG",
    "maybe_fake_quantize_nvfp4_weight_tensors",
    "nvfp4_fake_qat_amax_cache",
]
