"""The interface ``GpuDeltaLayout`` reads from a model mapping (CUDA receiver only)."""

from __future__ import annotations

from typing import Protocol

from miles.backends.sglang_utils.gpu_delta.bindings import (
    ComputedImage,
    ConsumerSnapshot,
    DerivedImage,
    ParameterBindings,
    TensorBinding,
)


class ModelMapping(Protocol):
    """Startup bindings and derived views for one SGLang model implementation.

    Constructed as ``cls(model, shadows)`` once per admission from the live model, where ``shadows``
    is the session's ``CanonicalShadows``, handed to ``ParameterBindings``; the layout releases the
    mapping after ``finish``.
    """

    # Dense and routed-expert binding context; the layout reads excluded, modules, moe_layers, the aliases a
    # mapping declares for fused views of one live allocation, and the shadowed tensors it fills from the
    # checkpoint at the first prepare.
    parameters: ParameterBindings
    # Live consumers pinned at describe time, while binding; their identity is rechecked at finish and
    # every apply. finish() may not append to them.
    consumers: list[ConsumerSnapshot]
    # DerivedImage copies and ComputedImages refreshed after the matrix batches; filled by finish.
    # They read only canonical-bound storage (live parameters or shadows), never a derived destination,
    # and write only module parameters or tensors a snapshot in consumers pins.
    derived: list[DerivedImage | ComputedImage]

    # Called for every canonical tensor the layout does not exclude as static. Returns the binding, or
    # None after recording a reason in parameters.excluded; never excludes a tensor it does not
    # recognize (that fails as an unclassified canonical tensor). Raises _DtypeMismatch (collected)
    # or ValueError. May normalize meta in place; the plan digest is checked against the normalized meta.
    # The binding applies meta["encoding"], the plan's canonical_encoding; the layout rejects any other.
    def bind(self, name: str, meta: dict) -> TensorBinding | None: ...

    # Runs once at the first prepare, after the layout's identity check.
    def finish(self) -> None: ...
