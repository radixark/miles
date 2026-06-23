"""Unified packed-sequence layout handling for the FSDP backend.

One registry + one boundary derivation behind which each stateful arch registers how it resets
per-document state under THD packing. The actor calls ``apply_packing`` once per lifetime
(config-time class patches; post-load instance patches) instead of per-arch if-chains, and new
archs plug in by registering a ``PackingPatch`` in ``specs/`` (or registering nothing when the
arch packs natively). See ``boundaries.PackedSeqContext`` for the shared cu_seqlens/seq_idx.
"""

from .boundaries import PackedSeqContext, packed_seq_context
from .registry import PackingPatch, apply_packing, get_packing_patches, register_packing_patch

__all__ = [
    "PackedSeqContext",
    "packed_seq_context",
    "PackingPatch",
    "register_packing_patch",
    "get_packing_patches",
    "apply_packing",
]
