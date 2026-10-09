"""Per-token log-prob and entropy over vocab-parallel logits, without a vocab-sized buffer.

Each selected row keeps a few fp32 numbers measured from its max ``m``, with ``d = (x - m) / T``:
``log S`` for ``S = sum exp(d)``, the target's ``d_y`` and, for entropy, the softmax mean ``mu`` of
``d``. Then ``log p = d_y - log S`` and ``H = log S - mu``, which keeps a confident token as accurate
as ``torch.log_softmax``, where ``target_logit - lse`` would cancel. The backward writes
``(g * (onehot(y) - p) - c * p * (d - mu)) / T`` with ``1 - p`` at the target from ``expm1``.
No-grad passes call the same function, so their log-probs equal the training forward's bit for bit
up to the order NCCL adds more than two shards.
"""

from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor

from miles.kernels.softmax import token_log_softmax as kernels


def fused_log_probs_and_entropy(
    logits: Tensor,
    rows: Tensor,
    targets: Tensor,
    *,
    tp_group: dist.ProcessGroup | None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
    inplace_backward: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Fp32 ``[R]`` log-probs of ``targets`` (and entropies) at ``rows`` of CUDA ``logits / temperature``.

    ``logits`` is ``[N, V / TP]``, split over ``tp_group``; ``rows`` are unique and ``targets`` are
    below ``vocab_size``, past which columns are vocab padding outside the softmax. ``inplace_backward``
    writes the gradient into ``logits``, so nothing may read them after this op's backward.
    """
    if not logits.is_cuda:
        raise ValueError(f"the fused log-prob op needs CUDA logits, got {logits.device}")
    assert logits.dim() == 2 and logits.stride(-1) == 1, f"need [N, V] logits, got {tuple(logits.shape)}"
    assert rows.shape == targets.shape, f"{tuple(rows.shape)} rows vs {tuple(targets.shape)} targets"
    temperature = float(temperature) if temperature > 0 else 1.0

    log_probs, entropy = _FusedLogProbsAndEntropy.apply(
        logits,
        rows.long(),
        targets.long(),
        tp_group,
        _unpadded_columns(logits, tp_group, vocab_size),
        temperature,
        with_entropy,
        with_entropy and entropy_requires_grad,
        inplace_backward,
    )
    return log_probs, (entropy if with_entropy else None)


class _RowStats(NamedTuple):
    """Per selected row, over the true vocabulary, with ``d = (x - m) / T`` for the row max ``m``."""

    max: Tensor  # m
    log_sum: Tensor  # log sum_v exp(d_v)
    target: Tensor  # d_y
    mean: Tensor | None  # sum_v p_v d_v, only with entropy


class _FusedLogProbsAndEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        rows,
        targets,
        tp_group,
        n_unpadded_cols,
        temperature,
        with_entropy,
        entropy_requires_grad,
        inplace_backward,
    ):
        stats = _row_statistics(logits, rows, targets, tp_group, n_unpadded_cols, temperature, with_entropy)
        log_probs = stats.target - stats.log_sum
        entropy = stats.log_sum - stats.mean if with_entropy else log_probs.new_empty(0)
        if not entropy_requires_grad:
            ctx.mark_non_differentiable(entropy)
        ctx.save_for_backward(
            logits, rows, targets, stats.max, stats.log_sum, log_probs, stats.mean if entropy_requires_grad else None
        )
        ctx.tp_group = tp_group
        ctx.n_unpadded_cols = n_unpadded_cols
        ctx.temperature = temperature
        ctx.entropy_requires_grad = entropy_requires_grad
        ctx.inplace_backward = inplace_backward
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        logits, rows, targets, row_max, log_sum, log_probs, mean = ctx.saved_tensors
        grad = logits if ctx.inplace_backward else torch.empty_like(logits)
        kernels.write_logits_grad(
            grad,
            logits,
            rows,
            targets,
            row_max,
            log_sum,
            mean,
            -torch.expm1(log_probs),  # 1 - p at the target, exact even when p is close to 1
            _contiguous_or_none(grad_log_probs),
            _contiguous_or_none(grad_entropy) if ctx.entropy_requires_grad else None,
            vocab_start=_vocab_start(logits, ctx.tp_group),
            n_unpadded_cols=ctx.n_unpadded_cols,
            temperature=ctx.temperature,
        )
        if rows.numel() < grad.size(0):  # rows are unique, so otherwise every row was scored
            kernels.zero_unscored_rows(grad, rows)
        if ctx.inplace_backward:
            # the kernels write through raw pointers; the bump makes a second reader fail, not read the grad
            torch.autograd.graph.increment_version(logits)
        return grad, None, None, None, None, None, None, None, None


def _row_statistics(logits, rows, targets, tp_group, n_unpadded_cols, temperature, with_entropy) -> _RowStats:
    """Statistics of the selected rows, combined over the vocab-parallel group."""
    vocab_start = _vocab_start(logits, tp_group)
    row_max, row_sum, row_dsum, target = kernels.row_statistics(
        logits,
        rows,
        targets,
        vocab_start=vocab_start,
        n_unpadded_cols=n_unpadded_cols,
        temperature=temperature,
        with_entropy=with_entropy,
    )
    in_shard = (targets >= vocab_start) & (targets < vocab_start + n_unpadded_cols)
    row_max, row_sum, row_dsum, target = _combine_over_tp(
        row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature
    )
    return _RowStats(row_max, torch.log(row_sum), target, (row_dsum / row_sum if with_entropy else None))


def _combine_over_tp(row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature):
    """Re-measure each shard's sums from the global max and add them; only the target's shard adds ``d_y``."""
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return row_max, row_sum, row_dsum, target
    global_max = row_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    # an all-padding shard has max -inf and zero sums; a -inf shift would make -inf * 0 = nan
    shift = torch.where(row_sum > 0, (row_max - global_max) / temperature, 0.0)
    rescale = torch.exp(shift)
    columns = [row_sum * rescale, torch.where(in_shard, target + shift, torch.zeros_like(target))]
    if row_dsum is not None:
        columns.append((row_dsum + shift * row_sum) * rescale)
    sums = torch.stack(columns, dim=1)
    dist.all_reduce(sums, group=tp_group)
    return global_max, sums[:, 0], (sums[:, 2] if row_dsum is not None else None), sums[:, 1]


def _vocab_start(logits: Tensor, tp_group: dist.ProcessGroup | None) -> int:
    """First vocab id of this rank's shard: Megatron splits the padded vocab into equal shards."""
    if tp_group is None:
        return 0
    return dist.get_rank(tp_group) * logits.size(1)


def _unpadded_columns(logits: Tensor, tp_group: dist.ProcessGroup | None, vocab_size: int | None) -> int:
    """How many of this shard's columns are in the true vocabulary; the rest are padding."""
    if vocab_size is None:
        return logits.size(1)
    return min(max(vocab_size - _vocab_start(logits, tp_group), 0), logits.size(1))


def _contiguous_or_none(tensor: Tensor | None) -> Tensor | None:
    return tensor.contiguous() if tensor is not None else None
