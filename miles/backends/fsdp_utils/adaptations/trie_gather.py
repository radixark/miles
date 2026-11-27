"""Row gathers over a trie row whose backward sums a shared row's copies in fp32.

A token the samples share is stored once in a trie row, and gathered once for every segment that reads it, so its
gradient is a sum over those copies. ``index_select``'s own backward adds the copies in the source dtype, rounding
to bf16 after each one, which leaves a trie row's gradients measurably less accurate than standard packing's.
"""

import torch


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(rows)
        ctx.source_shape = source.shape
        return source.index_select(0, rows)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (rows,) = ctx.saved_tensors
        total = grad.new_zeros(ctx.source_shape, dtype=torch.promote_types(grad.dtype, torch.float32))
        return total.index_add_(0, rows, grad.to(total.dtype)).to(grad.dtype), None


def gather_rows(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``source.index_select(0, rows)``, with a backward that sums each source row's copies in fp32."""
    return _GatherRows.apply(source, rows)
