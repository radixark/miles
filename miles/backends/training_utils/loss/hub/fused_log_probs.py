"""Log-probs of target tokens and entropy over vocab-parallel logits, without a vocab-sized buffer.

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
    """Fp32 log-probs of ``targets`` (and ``[R]`` entropies) at ``rows`` of CUDA ``logits / temperature``.

    ``logits`` is ``[N, V / TP]``, split over ``tp_group``; ``rows`` are unique. ``targets`` is ``[R]``
    or ``[R, K]`` and the log-probs take its shape; a ``-1`` target is padding, with log-prob 0 and no
    gradient, and the rest are below ``vocab_size``, past which columns are vocab padding outside the
    softmax. ``inplace_backward`` writes the gradient into ``logits``, so nothing may read them after
    this op's backward.
    """
    if not logits.is_cuda:
        raise ValueError(f"the fused log-prob op needs CUDA logits, got {logits.device}")
    assert logits.dim() == 2 and logits.stride(-1) == 1, f"need [N, V] logits, got {tuple(logits.shape)}"
    assert rows.dim() == 1 and targets.shape[:1] == rows.shape, f"{tuple(rows.shape)} rows vs {tuple(targets.shape)}"
    temperature = float(temperature) if temperature > 0 else 1.0

    log_probs, entropy = _FusedLogProbsAndEntropy.apply(
        logits,
        rows.long(),
        (targets if targets.dim() == 2 else targets.unsqueeze(1)).long(),
        tp_group,
        _unpadded_columns(logits, tp_group, vocab_size),
        temperature,
        with_entropy,
        with_entropy and entropy_requires_grad,
        inplace_backward,
    )
    return log_probs.view(targets.shape), (entropy if with_entropy else None)


class _RowStats(NamedTuple):
    """Per selected row, over the true vocabulary, with ``d = (x - m) / T`` for the row max ``m``."""

    max: Tensor  # m
    log_sum: Tensor  # log sum_v exp(d_v)
    target: Tensor  # d_t per target, [R, K]
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
        log_probs = torch.where(targets >= 0, stats.target - stats.log_sum.unsqueeze(1), 0.0)
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
        if grad_log_probs is not None:
            grad_log_probs = grad_log_probs.reshape(targets.shape).masked_fill(targets < 0, 0.0).contiguous()
        kernels.write_logits_grad(
            grad,
            logits,
            rows,
            targets,
            log_probs,
            -torch.expm1(log_probs),  # 1 - p at each target, exact even when p is close to 1
            row_max,
            log_sum,
            mean,
            grad_log_probs,
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


def fused_support_log_probs(
    logits: Tensor,
    rows: Tensor,
    targets: Tensor,
    support: Tensor,
    *,
    tp_group: dist.ProcessGroup | None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    with_entropy: bool = False,
    entropy_over_support: bool = False,
    entropy_requires_grad: bool = True,
    inplace_backward: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Like ``fused_log_probs_and_entropy``, but normalized over a per-row support, not the vocabulary.

    ``support`` is ``[R, S]`` token ids, unique within a row, ``-1`` padding; every target lies in its
    row's support. Returns the targets' log-probs (in ``targets``' shape), the ``[R, S]`` log-probs of
    the support ids themselves, and the entropy, over the vocabulary unless ``entropy_over_support``.
    A row with no support scores 0, with no gradient. Only the support logits are read, plus one
    pass over the vocabulary when the entropy is over it.
    """
    if not logits.is_cuda:
        raise ValueError(f"the fused log-prob op needs CUDA logits, got {logits.device}")
    assert logits.dim() == 2 and logits.stride(-1) == 1, f"need [N, V] logits, got {tuple(logits.shape)}"
    assert (
        support.dim() == 2 and support.shape[:1] == rows.shape == targets.shape[:1]
    ), f"{tuple(rows.shape)} rows vs {tuple(targets.shape)} targets vs {tuple(support.shape)} support"
    temperature = float(temperature) if temperature > 0 else 1.0
    target_log_probs, support_log_probs, entropy = _FusedSupportLogProbs.apply(
        logits,
        rows.long(),
        (targets if targets.dim() == 2 else targets.unsqueeze(1)).long(),
        support.long(),
        tp_group,
        _unpadded_columns(logits, tp_group, vocab_size),
        temperature,
        with_entropy,
        with_entropy and entropy_over_support,
        with_entropy and entropy_requires_grad,
        inplace_backward,
    )
    return target_log_probs.view(targets.shape), support_log_probs, (entropy if with_entropy else None)


class _FusedSupportLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        rows,
        targets,
        support,
        tp_group,
        n_unpadded_cols,
        temperature,
        with_entropy,
        entropy_over_support,
        entropy_requires_grad,
        inplace_backward,
    ):
        n_targets = targets.size(1)
        listed = torch.cat([targets, support], dim=1)
        in_shard, listed_logits = _gather_listed(logits, rows, listed, tp_group, n_unpadded_cols)
        support_in_shard, support_logits = in_shard[:, n_targets:], listed_logits[:, n_targets:]
        # the support's own statistics on this shard, as the kernel computes them over the vocabulary
        support_max = torch.where(support_in_shard, support_logits, -torch.inf).amax(dim=1)
        d = torch.where(
            support_in_shard, (support_logits - support_max.unsqueeze(1)) * (1.0 / temperature), -torch.inf
        )
        e = d.exp()
        support_sum = e.sum(dim=1)
        support_dsum = torch.where(support_in_shard, e * d, 0.0).sum(dim=1) if entropy_over_support else None
        listed_d = torch.where(in_shard, (listed_logits - support_max.unsqueeze(1)) * (1.0 / temperature), 0.0)
        support_max, support_sum, support_dsum, listed_d = _combine_over_tp(
            support_max, support_sum, support_dsum, listed_d, in_shard, tp_group, temperature
        )
        has_support = (support >= 0).any(dim=1, keepdim=True)
        support_log_sum = torch.log(support_sum)
        listed_log_probs = torch.where((listed >= 0) & has_support, listed_d - support_log_sum.unsqueeze(1), 0.0)
        target_log_probs, support_log_probs = listed_log_probs[:, :n_targets], listed_log_probs[:, n_targets:]

        vocab_stats = None
        if with_entropy and entropy_over_support:
            entropy = torch.where(has_support.squeeze(1), support_log_sum - support_dsum / support_sum, 0.0)
        elif with_entropy:
            vocab_stats = _row_statistics(logits, rows, targets[:, :0], tp_group, n_unpadded_cols, temperature, True)
            entropy = vocab_stats.log_sum - vocab_stats.mean
        else:
            entropy = target_log_probs.new_empty(0)
        if not entropy_requires_grad:
            ctx.mark_non_differentiable(entropy)
        ctx.save_for_backward(
            logits,
            rows,
            targets,
            support,
            target_log_probs,
            support_log_probs,
            support_max,
            support_log_sum,
            support_dsum / support_sum if entropy_requires_grad and entropy_over_support else None,
            *(vocab_stats.max, vocab_stats.log_sum, vocab_stats.mean) if vocab_stats is not None else (None,) * 3,
        )
        ctx.tp_group = tp_group
        ctx.n_unpadded_cols = n_unpadded_cols
        ctx.temperature = temperature
        ctx.entropy_requires_grad = entropy_requires_grad
        ctx.inplace_backward = inplace_backward
        return target_log_probs, support_log_probs, entropy

    @staticmethod
    def backward(ctx, grad_target_log_probs, grad_support_log_probs, grad_entropy):
        (
            logits,
            rows,
            targets,
            support,
            target_log_probs,
            support_log_probs,
            support_max,
            support_log_sum,
            support_mean,
            vocab_max,
            vocab_log_sum,
            vocab_mean,
        ) = ctx.saved_tensors
        grad = logits if ctx.inplace_backward else torch.empty_like(logits)
        grad_entropy = grad_entropy.contiguous() if grad_entropy is not None and ctx.entropy_requires_grad else None
        vocab_entropy_grad = grad_entropy if vocab_mean is not None else None
        # outside the support only a vocabulary entropy has a gradient; the dense pass writes it, or zeros
        kernels.write_logits_grad(
            grad,
            logits,
            rows,
            None,
            None,
            None,
            vocab_max if vocab_max is not None else support_max,
            vocab_log_sum if vocab_log_sum is not None else support_log_sum,
            vocab_mean,
            None,
            vocab_entropy_grad,
            vocab_start=_vocab_start(logits, ctx.tp_group),
            n_unpadded_cols=ctx.n_unpadded_cols,
            temperature=ctx.temperature,
        )
        support_grad = _support_column_grads(
            targets,
            support,
            target_log_probs,
            support_log_probs,
            grad_target_log_probs,
            grad_support_log_probs,
            support_max,
            support_log_sum,
            support_mean,
            grad_entropy if support_mean is not None else None,
            (vocab_max, vocab_log_sum, vocab_mean, vocab_entropy_grad) if vocab_entropy_grad is not None else None,
            ctx.temperature,
        )
        local_cols = support - _vocab_start(logits, ctx.tp_group)
        owned = (support >= 0) & (local_cols >= 0) & (local_cols < ctx.n_unpadded_cols)
        # support ids are unique within a row, so this plain assignment writes each column once
        grad[rows.unsqueeze(1).expand_as(support)[owned], local_cols[owned]] = support_grad[owned].to(grad.dtype)
        if rows.numel() < grad.size(0):
            kernels.zero_unscored_rows(grad, rows)
        if ctx.inplace_backward:
            torch.autograd.graph.increment_version(logits)
        return grad, None, None, None, None, None, None, None, None, None, None


def _support_column_grads(
    targets,
    support,
    target_log_probs,
    support_log_probs,
    grad_target_log_probs,
    grad_support_log_probs,
    support_max,
    support_log_sum,
    support_mean,
    support_entropy_grad,
    vocab_entropy,
    temperature,
) -> Tensor:
    """``[R, S]`` logits gradient at the support columns, from the saved log-probs alone."""
    valid = support >= 0
    coefficient = torch.zeros_like(support_log_probs)
    if grad_support_log_probs is not None:
        coefficient = coefficient + grad_support_log_probs.masked_fill(~valid, 0.0)
    if grad_target_log_probs is not None:
        g = grad_target_log_probs.reshape(targets.shape).masked_fill(targets < 0, 0.0)
        # a target adds its gradient at the support column holding the same token
        coefficient = coefficient + ((support.unsqueeze(2) == targets.unsqueeze(1)) * g.unsqueeze(1)).sum(dim=2)
    p = support_log_probs.exp()
    total = coefficient.sum(dim=1, keepdim=True)
    # 1 - p from expm1: exact for the confident support token
    grad = coefficient * -torch.expm1(support_log_probs) - (total - coefficient) * p
    d = support_log_probs + support_log_sum.unsqueeze(1)  # (x - support max) / T
    if support_entropy_grad is not None:
        grad = grad - support_entropy_grad.unsqueeze(1) * p * (d - support_mean.unsqueeze(1))
    if vocab_entropy is not None:
        vocab_max, vocab_log_sum, vocab_mean, vocab_entropy_grad = vocab_entropy
        vocab_d = d + ((support_max - vocab_max) / temperature).unsqueeze(1)
        vocab_p = (vocab_d - vocab_log_sum.unsqueeze(1)).exp()
        grad = grad - vocab_entropy_grad.unsqueeze(1) * vocab_p * (vocab_d - vocab_mean.unsqueeze(1))
    has_support = valid.any(dim=1, keepdim=True)
    return torch.where(valid & has_support, grad / temperature, 0.0)


def _gather_listed(logits, rows, listed, tp_group, n_unpadded_cols) -> tuple[Tensor, Tensor]:
    """Which listed ids this shard holds, and their fp32 logits (garbage where it does not)."""
    local_cols = listed - _vocab_start(logits, tp_group)
    in_shard = (listed >= 0) & (local_cols >= 0) & (local_cols < n_unpadded_cols)
    return in_shard, logits[rows.unsqueeze(1), local_cols.clamp(0, logits.size(1) - 1)].float()


def _row_statistics(logits, rows, targets, tp_group, n_unpadded_cols, temperature, with_entropy) -> _RowStats:
    """Statistics of the selected rows, combined over the vocab-parallel group."""
    row_max, row_sum, row_dsum = kernels.row_statistics(
        logits, rows, n_unpadded_cols=n_unpadded_cols, temperature=temperature, with_entropy=with_entropy
    )
    in_shard, target_logits = _gather_listed(logits, rows, targets, tp_group, n_unpadded_cols)
    target = torch.where(in_shard, (target_logits - row_max.unsqueeze(1)) * (1.0 / temperature), 0.0)
    row_max, row_sum, row_dsum, target = _combine_over_tp(
        row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature
    )
    return _RowStats(row_max, torch.log(row_sum), target, (row_dsum / row_sum if with_entropy else None))


def _combine_over_tp(row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature):
    """Re-measure each shard's sums from the global max and add them; only a target's shard adds its ``d``."""
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return row_max, row_sum, row_dsum, target
    global_max = row_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    # an all-padding shard has max -inf and zero sums; a -inf shift would make -inf * 0 = nan
    shift = torch.where(row_sum > 0, (row_max - global_max) / temperature, 0.0)
    rescale = torch.exp(shift)
    columns = [(row_sum * rescale).unsqueeze(1), torch.where(in_shard, target + shift.unsqueeze(1), 0.0)]
    if row_dsum is not None:
        columns.append(((row_dsum + shift * row_sum) * rescale).unsqueeze(1))
    sums = torch.cat(columns, dim=1)
    dist.all_reduce(sums, group=tp_group)
    n_targets = target.size(1)
    combined_dsum = sums[:, 1 + n_targets] if row_dsum is not None else None
    return global_max, sums[:, 0], combined_dsum, sums[:, 1 : 1 + n_targets]


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
