"""Deferred TOP LM head: the model returns the LM head's input rows; the TOP loss computes logits in row chunks.

Full-sequence logits are the largest long-context buffer on the last pipeline stage (32k tokens x 248k vocabulary is
16 GiB in BF16, and backward allocates the same again). The LM head is a row-wise batch-invariant GEMM, so a chunk's
logits are the bytes the full GEMM would give those rows, and the scores stay serving's.
"""

from __future__ import annotations

import torch


class TopLMHeadRows:
    """``[rows, V]`` logits held as the LM head's input rows ``hidden`` [rows, H] and its ``weight`` [V, H]."""

    def __init__(self, hidden: torch.Tensor, weight: torch.Tensor):
        self.hidden, self.weight = hidden, weight

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.hidden.shape[0], self.weight.shape[0]))

    @property
    def dtype(self) -> torch.dtype:
        return self.hidden.dtype

    @property
    def device(self) -> torch.device:
        return self.hidden.device

    def dim(self) -> int:
        return 2

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def __getitem__(self, rows) -> TopLMHeadRows:
        if not isinstance(rows, slice):
            raise TypeError("[top] deferred logits support row slices only")
        return TopLMHeadRows(self.hidden[rows], self.weight)

    def split(self, sizes: list[int]) -> tuple[TopLMHeadRows, ...]:
        """Consecutive row slices of the given sizes, as ``Tensor.split`` gives a logits tensor's rows."""
        return tuple(TopLMHeadRows(hidden, self.weight) for hidden in self.hidden.split(sizes))

    def sum(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """A scalar on the autograd graph of the rows and the LM head weight, for the ``0 * logits.sum()`` terms that
        keep every rank's backward traversing the graph; not the logits' sum (never materialized)."""
        return self.hidden.sum(dtype=dtype) + self.weight.sum(dtype=dtype)

    def logits(self, lo: int, hi: int) -> torch.Tensor:
        """Rows lo:hi of the logits, bitwise the output layer's (serving's batch-invariant GEMM)."""
        from sglang.srt.batch_invariant_ops import matmul_persistent

        return matmul_persistent(self.hidden[lo:hi].contiguous(), self.weight.t())


class TopDeferredLogits(TopLMHeadRows):
    """The model output ``[1, T, V]`` in packed THD layout."""

    @property
    def shape(self) -> torch.Size:
        return torch.Size((1, *super().shape))

    def dim(self) -> int:
        return 3

    def squeeze(self, dim: int) -> TopLMHeadRows:
        if dim != 0:
            raise ValueError("[top] deferred logits squeeze only the microbatch dimension")
        return TopLMHeadRows(self.hidden, self.weight)


def deferred_logits_processor(rows: torch.Tensor, select=None):
    """GPTModel ``output_processor``: keep the final-norm rows ``rows`` (packed layout order), defer the LM head.
    ``select`` maps the [S, H] decoder rows to those rows instead (e.g. through a context-parallel all-gather)."""

    def output_processor(*, hidden_states, output_layer, output_weight, **_):
        weight = output_layer.weight if output_weight is None else output_weight
        if hidden_states.dim() != 3 or hidden_states.shape[1] != 1:
            raise ValueError("[top] deferred LM head expects [S, 1, H] decoder output")
        flat = hidden_states[:, 0]
        return TopDeferredLogits(select(flat) if select is not None else flat.index_select(0, rows), weight)

    return output_processor
