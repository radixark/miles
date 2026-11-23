"""Validate the delivered row index before slot-order reduction."""

from __future__ import annotations
import os
from dataclasses import dataclass
import torch

# Transport identity checks: annotated slot rows and SGLang's device-side receive/return receipts, whose failure bits
# are read without a host sync (slot_order_deepep.read_pending_receipts). Off, the slot dispatch moves only its handle
# and the per-step exactness audit stands in for them.
CHECKS = os.environ.get("TOP_TRANSPORT_CHECKS", "1") != "0"


@dataclass(frozen=True)
class TransportReceipt:
    input_index: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def active(self) -> torch.Tensor:
        return self.topk_ids >= 0


def validate_receipt(receipt: TransportReceipt, *, num_rows: int) -> None:
    """Shape of the receipt. Its contents are checked where they are produced: the slot transport returns slot (t, k)
    as row t * topk + k by construction, and its device-side receipts check the returned rows' expert ids and FP32
    weight bytes against the router's."""
    idx, ids, weights = (receipt.input_index, receipt.topk_ids, receipt.topk_weights)
    if not idx.shape == ids.shape == weights.shape:
        raise ValueError(f"[top] receipt shapes disagree: {idx.shape} {ids.shape} {weights.shape}")
    if idx.dim() != 2:
        raise ValueError(f"[top] receipt must be [num_tokens, topk], got {tuple(idx.shape)}")
    if idx.numel() > num_rows:
        raise ValueError(f"[top] receipt addresses {idx.numel()} slots, the buffer holds {num_rows} rows")
