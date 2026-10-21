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


def _softmax_stats(logits, rows, targets, options: _FusedOptions) -> kernels.SoftmaxStats:
    """Stats of the selected rows over the whole true vocabulary, combined over the vocab-parallel group."""
    logit_max, sum_exp, sum_exp_logit = kernels.row_statistics(
        logits,
        rows,
        vocab=options.vocab,
        temperature=options.temperature,
        with_entropy=options.with_entropy,
    )
    local_cols = targets - options.vocab.start
    in_shard = options.vocab.holds(targets)
    raw_target_logits = logits[rows.unsqueeze(1), local_cols.clamp(0, logits.size(1) - 1)].float()
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
