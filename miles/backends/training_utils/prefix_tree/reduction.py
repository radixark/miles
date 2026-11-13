"""Fixed-order shared-row gradient reduction, without CUDA scatter atomics."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


class _OrderedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices, order, lengths):
        ctx.save_for_backward(order, lengths)
        ctx.input_dtype = value.dtype
        return value.index_select(0, indices)

    @staticmethod
    def backward(ctx, gradient):
        order, lengths = ctx.saved_tensors
        dtype = torch.float64 if ctx.input_dtype == torch.float64 else torch.float32
        ordered = gradient.index_select(0, order).to(dtype)
        # Each output row owns one segment. No threads atomically update the
        # same parent; contributions are summed in their prepared stable order.
        reduced = torch.segment_reduce(ordered, reduce="sum", lengths=lengths)
        return reduced.to(ctx.input_dtype), None, None, None


@dataclass(frozen=True)
class OrderedRowGather:
    """Prepare once on CPU, then reuse on device for forward/backward.

    Deterministic for a fixed occurrence layout. Changing the tree can change
    the contributions themselves (kernel shapes/rounding), not just their order.
    """

    indices: torch.Tensor
    order: torch.Tensor
    lengths: torch.Tensor

    @classmethod
    def from_indices(cls, indices: Sequence[int], input_rows: int, *, device) -> "OrderedRowGather":
        if input_rows < 1 or any(index < 0 or index >= input_rows for index in indices):
            raise ValueError("row indices must address the input tensor")
        # Stable counting sort: row IDs are dense integers, so preparation is
        # O(gathered rows + input rows), rather than a comparison sort.
        occurrences = [[] for _ in range(input_rows)]
        for position, index in enumerate(indices):
            occurrences[index].append(position)
        lengths = [len(row) for row in occurrences]
        order = [position for row in occurrences for position in row]
        return cls(
            indices=torch.tensor(indices, dtype=torch.long, device=device),
            order=torch.tensor(order, dtype=torch.long, device=device),
            lengths=torch.tensor(lengths, dtype=torch.long, device=device),
        )

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return _OrderedGather.apply(value, self.indices, self.order, self.lengths)
