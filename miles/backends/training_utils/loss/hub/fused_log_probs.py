"""Fused per-token log-probability and entropy over vocab-parallel logits.

This op streams each selected row once and keeps a few fp32 numbers per row, all measured from the
row max ``m``: with ``d = (x - m) / T``, ``log S`` for ``S = sum exp(d)``, the target's ``d_y`` and,
for entropy, the softmax mean ``mu`` of ``d``. Then

    log p = d_y - log S        H = log S - mu

which keeps a confident token as accurate as ``torch.log_softmax`` (``target_logit - lse`` would
lose it to cancellation). Tensor-parallel ranks combine the statistics with one max and one sum
all-reduce over ``[rows, 2 or 3]``. The backward streams the rows again and writes

    dlogits = (g * (onehot(y) - p) - c * p * (d - mu)) / T

for the upstream gradients ``g`` of the log-prob and ``c`` of the entropy, taking ``1 - p`` at the
target from ``expm1``. With ``vocab_size`` the softmax runs over the true vocabulary only: the
padding columns Megatron appends have probability zero and get a zero gradient. With ``inplace_backward`` it writes into the logits buffer and zeroes the rows
it did not score, so the backward allocates nothing of vocab size.

No-grad scoring passes (reference, old actor, teacher) call the same autograd function, so their
log-probs equal the training forward's bit for bit; above two tensor-parallel ranks, only up to the
order in which NCCL adds the shards.

The kernels are in ``miles.kernels.softmax.token_log_softmax``; the op takes CUDA logits only.
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
    """Log-prob of ``targets`` and entropy at the selected ``rows`` of ``logits / temperature``.

    Args:
        logits: ``[N, V_local]`` vocab-parallel logits in any float dtype, last dim contiguous.
        rows: ``[R]`` int64 row indices into ``logits``, each at most once.
        targets: ``[R]`` int64 token ids in the full vocabulary, each below ``vocab_size``.
        tp_group: the vocab-parallel group; ``None`` for an unsplit vocabulary.
        vocab_size: the true vocabulary size; columns from it on are padding and excluded from the
            softmax. ``None`` counts every column.
        temperature: divides the logits; values ``<= 0`` mean no scaling, as in the torch path.
        with_entropy: also return the entropy of each row.
        entropy_requires_grad: when False the entropy is a metric and carries no gradient.
        inplace_backward: write the logits gradient into ``logits`` itself. Only safe when nothing
            else reads the logits after this op's backward.

    Returns:
        ``(log_probs, entropy)`` fp32 ``[R]`` tensors; ``entropy`` is None unless requested.
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
        _valid_columns(logits, tp_group, vocab_size),
        temperature,
        with_entropy,
        with_entropy and entropy_requires_grad,
        inplace_backward,
    )
    return log_probs, (entropy if with_entropy else None)


class _RowStats(NamedTuple):
    """Per selected row, over the whole vocabulary, with ``d = (x - m) / T`` for the row max ``m``."""

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
        n_valid,
        temperature,
        with_entropy,
        entropy_requires_grad,
        inplace_backward,
    ):
        stats = _row_statistics(logits, rows, targets, tp_group, n_valid, temperature, with_entropy)
        log_probs = stats.target - stats.log_sum
        entropy = stats.log_sum - stats.mean if with_entropy else log_probs.new_empty(0)
        if not entropy_requires_grad:
            ctx.mark_non_differentiable(entropy)
        ctx.save_for_backward(
            logits, rows, targets, stats.max, stats.log_sum, log_probs, stats.mean if entropy_requires_grad else None
        )
        ctx.tp_group = tp_group
        ctx.n_valid = n_valid
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
            n_valid=ctx.n_valid,
            temperature=ctx.temperature,
        )
        if rows.numel() < grad.size(0):  # rows are unique, so otherwise every row was scored
            kernels.zero_unscored_rows(grad, rows)
        if ctx.inplace_backward:
            # the kernels write through raw pointers; bump the version so a second reader of these
            # logits (e.g. a retained graph) fails loudly instead of reading the gradient
            torch.autograd.graph.increment_version(logits)
        return grad, None, None, None, None, None, None, None, None


def _row_statistics(logits, rows, targets, tp_group, n_valid, temperature, with_entropy) -> _RowStats:
    """Statistics of the selected rows, combined over the vocab-parallel group."""
    vocab_start = _vocab_start(logits, tp_group)
    row_max, row_sum, row_dsum, target = kernels.row_statistics(
        logits,
        rows,
        targets,
        vocab_start=vocab_start,
        n_valid=n_valid,
        temperature=temperature,
        with_entropy=with_entropy,
    )
    in_shard = (targets >= vocab_start) & (targets < vocab_start + n_valid)
    row_max, row_sum, row_dsum, target = _combine_over_tp(
        row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature
    )
    return _RowStats(row_max, torch.log(row_sum), target, (row_dsum / row_sum if with_entropy else None))


def _combine_over_tp(row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature):
    """Re-measure each shard's statistics from the global max and add them.

    Each shard reports ``sum exp(d)``, ``sum exp(d) * d`` and the target's ``d`` from its own max;
    moving the reference to the global max shifts every ``d`` by the same amount. Only the shard
    holding the target contributes its ``d_y``. A shard that is all padding reports a zero sum from
    a max of ``-inf`` and contributes nothing.
    """
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return row_max, row_sum, row_dsum, target
    global_max = row_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    # zero, not -inf, for an all-padding shard: its sums are zero, and -inf * 0 would be nan
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


def _valid_columns(logits: Tensor, tp_group: dist.ProcessGroup | None, vocab_size: int | None) -> int:
    """How many of this shard's columns are in the true vocabulary; the rest are padding."""
    if vocab_size is None:
        return logits.size(1)
    return min(max(vocab_size - _vocab_start(logits, tp_group), 0), logits.size(1))


def _contiguous_or_none(tensor: Tensor | None) -> Tensor | None:
    return tensor.contiguous() if tensor is not None else None
