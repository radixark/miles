"""Log-probs of target tokens and entropy over vocab-parallel logits, without a vocab-sized buffer.

Each selected row keeps a few fp32 numbers measured from its max ``m``, with ``d = (x - m) / T``:
``log S`` for ``S = sum exp(d)``, the target's ``d_y`` and, for entropy, the softmax mean ``mu`` of
``d``. Then ``log p = d_y - log S`` and ``H = log S - mu``, which keeps a confident token as accurate
as ``torch.log_softmax``, where ``target_logit - lse`` would cancel. The backward writes
``(g * (onehot(y) - p) - c * p * (d - mu)) / T`` with ``1 - p`` at the target from ``expm1``.
No-grad passes call the same function, so their log-probs equal the training forward's bit for bit
up to the order NCCL adds more than two shards.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from miles.backends.training_utils.loss.hub.math_utils import true_vocab_size
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
    assert temperature > 0, f"temperature {temperature}: resolve greedy (0) with math_utils.scaling_temperature"
    options = _FusedOptions(
        tp_group=tp_group,
        vocab=_vocab_range(logits, tp_group, vocab_size),
        temperature=float(temperature),
        with_entropy=with_entropy,
        entropy_requires_grad=with_entropy and entropy_requires_grad,
        inplace_backward=inplace_backward,
    )
    _check_ids_in_vocabulary(targets, min_id=-1, vocab=options.vocab)  # -1 is a padding target
    targets_2d = (targets if targets.dim() == 2 else targets.unsqueeze(1)).long()
    log_probs, entropy = _FusedLogProbsAndEntropy.apply(logits, rows.long(), targets_2d, options)
    return log_probs.view(targets.shape), (entropy if with_entropy else None)


@dataclass(frozen=True)
class _FusedOptions:
    """The op's non-tensor inputs."""

    tp_group: dist.ProcessGroup | None
    vocab: kernels.VocabRange
    temperature: float
    with_entropy: bool
    entropy_requires_grad: bool
    inplace_backward: bool


class _FusedLogProbsAndEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, rows, targets, options: _FusedOptions):
        stats = _softmax_stats(logits, rows, targets, options)
        log_probs = torch.where(targets >= 0, stats.target_logit - stats.log_sum_exp.unsqueeze(1), 0.0)
        entropy = stats.log_sum_exp - stats.expected_mean_logit if options.with_entropy else log_probs.new_empty(0)
        if not options.entropy_requires_grad:
            ctx.mark_non_differentiable(entropy)
            stats = stats._replace(expected_mean_logit=None)  # the backward needs it only for the entropy
        ctx.save_for_backward(logits, rows, targets, *stats)
        ctx.options = options
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        logits, rows, targets, *stats = ctx.saved_tensors
        options = ctx.options
        grad = logits if options.inplace_backward else torch.empty_like(logits)
        if grad_log_probs is not None:
            grad_log_probs = grad_log_probs.reshape(targets.shape).masked_fill(targets < 0, 0.0).contiguous()
        kernels.write_logits_grad(
            grad,
            logits,
            rows=rows,
            targets=targets,
            stats=kernels.SoftmaxStats(*stats),
            grad_log_probs=grad_log_probs,
            grad_entropy=_contiguous_or_none(grad_entropy) if options.entropy_requires_grad else None,
            vocab=options.vocab,
            temperature=options.temperature,
        )
        if rows.numel() < grad.size(0):  # rows are unique, so otherwise every row was scored
            kernels.zero_unscored_rows(grad, rows)
        if options.inplace_backward:
            # the kernels write through raw pointers; the bump makes a second reader fail, not read the grad
            torch.autograd.graph.increment_version(logits)
        return grad, None, None, None


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
    assert temperature > 0, f"temperature {temperature}: resolve greedy (0) with math_utils.scaling_temperature"
    options = _FusedOptions(
        tp_group=tp_group,
        vocab=_vocab_range(logits, tp_group, vocab_size),
        temperature=float(temperature),
        with_entropy=with_entropy,
        entropy_requires_grad=with_entropy and entropy_requires_grad,
        inplace_backward=inplace_backward,
    )
    for ids in (targets, support):
        _check_ids_in_vocabulary(ids, min_id=-1, vocab=options.vocab)
    targets_2d = (targets if targets.dim() == 2 else targets.unsqueeze(1)).long()
    target_log_probs, support_log_probs, entropy = _FusedSupportLogProbs.apply(
        logits, rows.long(), targets_2d, support.long(), options, with_entropy and entropy_over_support
    )
    return target_log_probs.view(targets.shape), support_log_probs, (entropy if with_entropy else None)


class _FusedSupportLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, rows, targets, support, options: _FusedOptions, entropy_over_support: bool):
        n_targets = targets.size(1)
        listed = torch.cat([targets, support], dim=1)
        in_shard, listed_logits = _gather_listed(logits, rows, listed, options.vocab)
        support_in_shard, support_logits = in_shard[:, n_targets:], listed_logits[:, n_targets:]
        # the support's own statistics on this shard, as the kernel computes them over the vocabulary
        masked_support = torch.where(support_in_shard, support_logits, -torch.inf)
        # amax has no identity: a rank with no response rows pads the support to width 0
        support_logit_max = (
            masked_support.amax(dim=1)
            if masked_support.size(1)
            else masked_support.new_full((masked_support.size(0),), -torch.inf)
        )
        inv_temperature = 1.0 / options.temperature
        d = torch.where(
            support_in_shard, (support_logits - support_logit_max.unsqueeze(1)) * inv_temperature, -torch.inf
        )
        e = d.exp()
        shard = kernels.ShardStats(
            support_logit_max,
            e.sum(dim=1),
            torch.where(support_in_shard, e * d, 0.0).sum(dim=1) if entropy_over_support else None,
            torch.where(in_shard, (listed_logits - support_logit_max.unsqueeze(1)) * inv_temperature, 0.0),
        )
        support_stats = _combine_over_tp(shard, in_shard, options.tp_group, options.temperature).as_softmax_stats()
        has_support = (support >= 0).any(dim=1, keepdim=True)
        listed_log_probs = torch.where(
            (listed >= 0) & has_support, support_stats.target_logit - support_stats.log_sum_exp.unsqueeze(1), 0.0
        )
        target_log_probs, support_log_probs = listed_log_probs[:, :n_targets], listed_log_probs[:, n_targets:]

        vocab_stats = None
        if options.with_entropy and entropy_over_support:
            support_entropy = support_stats.log_sum_exp - support_stats.expected_mean_logit
            entropy = torch.where(has_support.squeeze(1), support_entropy, 0.0)
        elif options.with_entropy:
            vocab_stats = _softmax_stats(logits, rows, targets[:, :0], options)
            entropy = vocab_stats.log_sum_exp - vocab_stats.expected_mean_logit
        else:
            entropy = target_log_probs.new_empty(0)
        if not options.entropy_requires_grad:
            ctx.mark_non_differentiable(entropy)
            # the backward needs the entropy means, and the vocabulary stats, only for the entropy
            support_stats, vocab_stats = support_stats._replace(expected_mean_logit=None), None
        # the backward reads the support's targets from the saved log-probs, not from its stats
        saved_stats = (*support_stats._replace(target_logit=None), *(vocab_stats or (None,) * 4))
        ctx.save_for_backward(logits, rows, targets, support, target_log_probs, support_log_probs, *saved_stats)
        ctx.options = options
        return target_log_probs, support_log_probs, entropy

    @staticmethod
    def backward(ctx, grad_target_log_probs, grad_support_log_probs, grad_entropy):
        logits, rows, targets, support, target_log_probs, support_log_probs, *saved_stats = ctx.saved_tensors
        support_stats = kernels.SoftmaxStats(*saved_stats[:4])
        vocab_stats = kernels.SoftmaxStats(*saved_stats[4:]) if saved_stats[4] is not None else None
        options = ctx.options
        grad = logits if options.inplace_backward else torch.empty_like(logits)
        grad_entropy = (
            grad_entropy.contiguous() if grad_entropy is not None and options.entropy_requires_grad else None
        )
        vocab_entropy_grad = grad_entropy if vocab_stats is not None else None
        if vocab_entropy_grad is None:
            # no gradient outside the support: zero every row with a store-only pass, reading no logits
            kernels.zero_unscored_rows(grad, rows[:0])
        else:
            # outside the support only the vocabulary entropy has a gradient; the dense pass writes it
            kernels.write_logits_grad(
                grad,
                logits,
                rows=rows,
                targets=None,
                stats=vocab_stats,
                grad_log_probs=None,
                grad_entropy=vocab_entropy_grad,
                vocab=options.vocab,
                temperature=options.temperature,
            )
            if rows.numel() < grad.size(0):
                kernels.zero_unscored_rows(grad, rows)
        support_grad = _support_column_grads(
            targets,
            support,
            target_log_probs,
            support_log_probs,
            grad_target_log_probs,
            grad_support_log_probs,
            support_stats,
            grad_entropy if support_stats.expected_mean_logit is not None else None,
            vocab_stats,
            vocab_entropy_grad,
            options.temperature,
        )
        local_cols = support - options.vocab.start
        owned = options.vocab.holds(support)
        # support ids are unique within a row, so this plain assignment writes each column once
        grad[rows.unsqueeze(1).expand_as(support)[owned], local_cols[owned]] = support_grad[owned].to(grad.dtype)
        if options.inplace_backward:
            torch.autograd.graph.increment_version(logits)
        return grad, None, None, None, None, None


def _support_column_grads(
    targets,
    support,
    target_log_probs,
    support_log_probs,
    grad_target_log_probs,
    grad_support_log_probs,
    support_stats: kernels.SoftmaxStats,
    support_entropy_grad,
    vocab_stats: kernels.SoftmaxStats | None,
    vocab_entropy_grad,
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
    d = support_log_probs + support_stats.log_sum_exp.unsqueeze(1)  # (x - support max) / T
    if support_entropy_grad is not None:
        grad = grad - support_entropy_grad.unsqueeze(1) * p * (d - support_stats.expected_mean_logit.unsqueeze(1))
    if vocab_entropy_grad is not None:
        vocab_d = d + ((support_stats.logit_max - vocab_stats.logit_max) / temperature).unsqueeze(1)
        vocab_p = (vocab_d - vocab_stats.log_sum_exp.unsqueeze(1)).exp()
        grad = grad - vocab_entropy_grad.unsqueeze(1) * vocab_p * (
            vocab_d - vocab_stats.expected_mean_logit.unsqueeze(1)
        )
    has_support = valid.any(dim=1, keepdim=True)
    return torch.where(valid & has_support, grad / temperature, 0.0)


def _gather_listed(logits, rows, listed, vocab: kernels.VocabRange) -> tuple[Tensor, Tensor]:
    """Which listed ids this shard holds, and their fp32 logits (garbage where it does not)."""
    local_cols = listed - vocab.start
    in_shard = vocab.holds(listed)
    return in_shard, logits[rows.unsqueeze(1), local_cols.clamp(0, logits.size(1) - 1)].float()


def _softmax_stats(logits, rows, targets, options: _FusedOptions) -> kernels.SoftmaxStats:
    """Stats of the selected rows over the whole true vocabulary, combined over the vocab-parallel group."""
    logit_max, sum_exp, sum_exp_logit = kernels.row_statistics(
        logits,
        rows,
        vocab=options.vocab,
        temperature=options.temperature,
        with_entropy=options.with_entropy,
    )
    in_shard, raw_target_logits = _gather_listed(logits, rows, targets, options.vocab)
    # a masked (-inf) target logit is read as -1e30, as the kernels read every logit
    target_logit = torch.where(
        in_shard, (raw_target_logits.clamp_min(-1.0e30) - logit_max.unsqueeze(1)) * (1.0 / options.temperature), 0.0
    )
    shard = kernels.ShardStats(logit_max, sum_exp, sum_exp_logit, target_logit)
    return _combine_over_tp(shard, in_shard, options.tp_group, options.temperature).as_softmax_stats()


def _combine_over_tp(shard: kernels.ShardStats, in_shard, tp_group, temperature) -> kernels.ShardStats:
    """Re-measure each shard's sums from the global max and add them; only a target's shard adds its ``d``."""
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return shard
    global_logit_max = shard.logit_max.clone()
    dist.all_reduce(global_logit_max, op=dist.ReduceOp.MAX, group=tp_group)
    # an all-padding shard has max -inf and zero sums; a -inf shift would make -inf * 0 = nan
    shift = torch.where(shard.sum_exp > 0, (shard.logit_max - global_logit_max) / temperature, 0.0)
    rescale = torch.exp(shift)
    columns = [
        (shard.sum_exp * rescale).unsqueeze(1),
        torch.where(in_shard, shard.target_logit + shift.unsqueeze(1), 0.0),
    ]
    if shard.sum_exp_logit is not None:
        columns.append(((shard.sum_exp_logit + shift * shard.sum_exp) * rescale).unsqueeze(1))
    sums = torch.cat(columns, dim=1)
    dist.all_reduce(sums, group=tp_group)
    n_targets = shard.target_logit.size(1)
    sum_exp_logit = sums[:, 1 + n_targets] if shard.sum_exp_logit is not None else None
    return kernels.ShardStats(global_logit_max, sums[:, 0], sum_exp_logit, sums[:, 1 : 1 + n_targets])


def _check_ids_in_vocabulary(ids: Tensor, *, min_id: int, vocab: kernels.VocabRange) -> None:
    """No shard holds an id past the vocabulary, so it would score a plausible wrong value instead of failing."""
    if not ids.numel():  # a rank with no response rows has nothing to check, and aminmax no identity
        return
    # the extremes, not a mask the size of ids; asserted on device, as a host check would sync the stream
    low, high = torch.aminmax(ids)
    torch._assert_async((low >= min_id) & (high < vocab.size), "fused log-prob token id outside the vocabulary")


def _vocab_range(logits: Tensor, tp_group: dist.ProcessGroup | None, vocab_size: int | None) -> kernels.VocabRange:
    """This rank's shard of the vocabulary: Megatron splits the padded vocab into equal shards."""
    tp_rank, tp_size = (dist.get_rank(tp_group), dist.get_world_size(tp_group)) if tp_group is not None else (0, 1)
    start = tp_rank * logits.size(1)
    size = true_vocab_size(vocab_size, logits.size(1), tp_size)
    return kernels.VocabRange(start, min(max(size - start, 0), logits.size(1)), size)


def _contiguous_or_none(tensor: Tensor | None) -> Tensor | None:
    return tensor.contiguous() if tensor is not None else None
