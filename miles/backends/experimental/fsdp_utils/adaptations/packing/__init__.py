"""Unified packed-sequence layout handling for the FSDP backend (one registry + one boundary derivation)."""

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
