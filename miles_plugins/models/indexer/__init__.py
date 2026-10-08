"""Shared pieces every sparse-attention indexer uses.

The indexer is frozen and its selection recomputed in the trainer on every
path; there is no replay.
"""

from miles_plugins.models.indexer.freeze import freeze_indexer_parameters
from miles_plugins.models.indexer.select import (
    EMPTY_SLOT,
    TOPK_BACKENDS,
    flashinfer_tie_break_value,
    get_indexer_topk_fn,
    select_indexer_topk,
)

__all__ = [
    "EMPTY_SLOT",
    "TOPK_BACKENDS",
    "flashinfer_tie_break_value",
    "freeze_indexer_parameters",
    "get_indexer_topk_fn",
    "select_indexer_topk",
]
