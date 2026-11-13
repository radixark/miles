"""Metadata shared by packed-document attention and recurrent layers."""

import torch


def document_ids(positions: torch.Tensor) -> torch.Tensor:
    """Return each token's document start as `arange(L) - positions`.

    Positions restart at zero in every document and the padding tail. The non-decreasing ids let attention skip
    blocks from other documents, and avoid the int64 cumsum that libtpu cannot lower with the scan emitter off.
    """
    token_index = torch.arange(positions.shape[-1], device=positions.device, dtype=torch.int32)
    return token_index - positions.to(torch.int32)
