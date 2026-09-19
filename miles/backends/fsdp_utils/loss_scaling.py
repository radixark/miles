"""Per-token loss normalization for the FSDP backend.

With ``--calculate-per-token-loss`` the shared loss wrapper returns an
unnormalized token-sum loss plus a token-count normalizer. Megatron's
gradient reduction divides the accumulated gradients by that count; the FSDP
backend has no such reduction and accumulates raw ``loss.backward()``
gradients instead, so its optimizer-step gradient lands ``dp_size /
global_num_tokens`` times the intended token mean (FSDP2 averages gradients
over the full mesh, i.e. by ``dp_size``). Scaling every micro-batch loss of
one optimizer step by ``dp_size / global_num_tokens`` restores the global
token-mean gradient.
"""

import logging

import torch
import torch.distributed as dist

from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def _num_tokens(loss_masks: list[torch.Tensor]) -> int | torch.Tensor:
    """Per-sample ``max(mask.sum(), 1)`` summed over one micro-batch.

    Mirrors the ``num_tokens`` normalizer of the shared loss wrapper
    (``miles/backends/training_utils/loss.py``), including counting a
    fully-masked sample as one token so it never yields a zero denominator.
    """
    return sum(torch.clamp_min(loss_mask.sum(), 1) for loss_mask in loss_masks)


def _scan_step_num_tokens(data_iterator: DataIterator, num_microbatches: list[int]) -> list[torch.Tensor]:
    """Token count of every optimizer step, read along the training schedule.

    Walks a second ``DataIterator`` sharing ``rollout_data`` with the training
    one (no tensor copies, and the training iterator's offset is untouched),
    so the counts describe exactly the micro-batches the training loop is
    about to consume.
    """
    scan = DataIterator(
        rollout_data=data_iterator.rollout_data,
        micro_batch_size=data_iterator.micro_batch_size,
        micro_batch_indices=data_iterator.micro_batch_indices,
    )
    device = data_iterator.rollout_data["loss_masks"][0].device
    step_counts = []
    for num_mb in num_microbatches:
        microbatch_counts = [_num_tokens(scan.get_next(["loss_masks"])["loss_masks"]) for _ in range(num_mb)]
        # keep a 0-dim tensor even for a step with no micro-batches so every step stacks
        step_counts.append(
            torch.stack(microbatch_counts).sum() if microbatch_counts else torch.zeros((), device=device)
        )
    return step_counts


def get_per_token_loss_scales(data_iterator: DataIterator, num_microbatches: list[int]) -> list[torch.Tensor]:
    """One loss scale per optimizer step: ``dp_size / global_num_tokens``.

    Call after the training data iterator is reset, so the scan replays the
    schedule the training loop is about to consume. A single SUM all-reduce
    over the data-parallel group shares every step's global token count, so
    all ranks scale every micro-batch of a step identically and the FSDP2
    gradient average lands on the global token mean.
    """
    if not num_microbatches:
        return []

    parallel_state = get_parallel_state()
    dp_size = parallel_state.intra_dp.size

    counts = torch.stack(_scan_step_num_tokens(data_iterator, num_microbatches))
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=parallel_state.intra_dp.group)

    return [dp_size / count for count in counts]
