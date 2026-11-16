"""The interface generic bindings and layout admission read from a kernel backend (CUDA receiver only)."""

from __future__ import annotations

from typing import Protocol

import torch

from miles.backends.sglang_utils.gpu_delta.byte_views import DestinationSegment
from miles.backends.sglang_utils.gpu_delta.derived import DerivedUpdate


class ExpertLayout(Protocol):
    """One routed-expert layer's live storage, as its SGLang kernel backend laid it out.

    A backend's ``identify_layout(layer, runner_backend)`` builds it once, from the warmed consumer,
    at the layer's first canonical tensor; ``ParameterBindings.moe_layers`` keeps it beside the layer
    for the session. It describes existing storage and never selects, builds or rewrites a consumer.
    Every method receives that same layer.
    """

    # Canonical format tag: bf16, nvfp4, mxfp4, mxfp8 or fp8-block128-ue8m0. Generic binding keys the
    # format's checkpoint rules on it (static NVFP4 activation calibration, NVFP4 global-scale targets,
    # block-128 wire normalization and scale suffix renames) and stores it as TensorBinding.format,
    # which also replaces the direct dtype check; a new tag therefore needs generic support.
    format: str
    # True when w13 holds gate and up halves; False when it holds one up projection.
    gated: bool

    # [start, stop) per canonical axis of one projection tensor, named by its checkpoint suffix, that
    # this rank's ordinary loader owns, including a padded loader's physical rank stride. Raises ValueError.
    def expert_slices(
        self, layer: torch.nn.Module, projection: str, suffix: str, shape: list[int]
    ) -> list[list[int]]: ...

    # Called only for an fp8-block128-ue8m0 weight_scale_inv tensor: the FP32 consumer image into which one
    # local projection's UE8M0 scales expand losslessly. Raises _DtypeMismatch (collected by the layout) or
    # ValueError; a backend that admits no block-128 format raises ValueError.
    def expert_scale_target(
        self, layer: torch.nn.Module, local_expert: int, projection: str, canonical_shape: tuple[int, int]
    ) -> torch.Tensor: ...

    # Disjoint rectangles of one local canonical projection and their live byte views, padding excluded,
    # in canonical_shape coordinates; suffix is "weight" or "weight_scale" after the generic renames.
    # Generic binding offsets the rectangles by the rank's slice origin and drops source_shape. Raises
    # _DtypeMismatch (collected) or ValueError.
    def expert_destinations(
        self,
        layer: torch.nn.Module,
        local_expert: int,
        projection: str,
        suffix: str,
        canonical_shape: tuple[int, int],
        canonical_dtype: str,
    ) -> tuple[DestinationSegment, ...]: ...

    # Consumer objects and tensors pinned at describe time, in a fixed order; their identity is rechecked
    # at finish and every apply.
    def consumer_objects(self, layer: torch.nn.Module) -> tuple: ...

    # Updates recomputing small consumer values from changed canonical scales, built once at the first
    # prepare after the identity and topology checks; empty when the format derives none. Raises ValueError.
    def expert_derived_updates(self, layer: torch.nn.Module) -> tuple[DerivedUpdate, ...]: ...
