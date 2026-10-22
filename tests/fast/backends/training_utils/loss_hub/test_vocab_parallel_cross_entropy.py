"""The torch-only vocab-parallel cross entropy that stands in for Megatron's fused one when it is absent."""

import sys

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.loss.hub import math_utils
from miles.backends.training_utils.loss.hub.math_utils import compute_log_probs, vocab_parallel_cross_entropy

SEQ, BATCH, VOCAB = 5, 2, 12


def _full_logits(dtype: torch.dtype) -> torch.Tensor:
    return (torch.randn(SEQ, BATCH, VOCAB, generator=torch.Generator().manual_seed(7)) * 4).to(dtype)


def _target() -> torch.Tensor:
    # Covers both shards of a TP=2 split, including each shard's first and last id.
    return torch.tensor([[0, 11], [5, 6], [6, 5], [11, 0], [3, 9]])


def _reference(full: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = full.detach().clone().requires_grad_()
    loss = -torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss.backward(_grad_output())
    return loss.detach(), logits.grad


def _grad_output() -> torch.Tensor:
    return torch.randn(SEQ, BATCH, generator=torch.Generator().manual_seed(11))


def _check_against_full_softmax(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(list(range(world_size)))
        shard = VOCAB // world_size
        for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 1e-2)):
            full = _full_logits(dtype)
            expected_loss, expected_grad = _reference(full, _target())
            local = full[..., rank * shard : (rank + 1) * shard].clone().requires_grad_()
            local_before = local.detach().clone()

            loss = vocab_parallel_cross_entropy(local, _target(), group)
            loss.backward(_grad_output())

            torch.testing.assert_close(loss, expected_loss, atol=tol, rtol=tol)
            assert local.grad.dtype == dtype
            torch.testing.assert_close(
                local.grad, expected_grad[..., rank * shard : (rank + 1) * shard], atol=tol, rtol=tol
            )
            torch.testing.assert_close(local.detach(), local_before, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_matches_the_full_vocabulary_log_softmax_on_every_shard(world_size: int) -> None:
    """Loss and gradient equal a plain log-softmax over the whole vocabulary, however it is split."""
    run_multiprocess(_check_against_full_softmax, world_size=world_size)


def _check_against_megatron(rank: int, world_size: int, port: int) -> None:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(list(range(world_size)))
        shard = VOCAB // world_size
        full = _full_logits(torch.bfloat16)
        results = []
        for fn in (vocab_parallel_cross_entropy, fused_vocab_parallel_cross_entropy):
            local = full[..., rank * shard : (rank + 1) * shard].clone().requires_grad_()
            loss = fn(local, _target(), group)
            loss.backward(_grad_output())
            results.append((loss.detach(), local.grad))
        (loss, grad), (megatron_loss, megatron_grad) = results
        torch.testing.assert_close(loss, megatron_loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(grad, megatron_grad, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_matches_megatron_fused_cross_entropy(world_size: int) -> None:
    """Where Megatron is installed, the stand-in computes what its fused kernel computes."""
    pytest.importorskip("megatron.core.fusions.fused_cross_entropy")
    run_multiprocess(_check_against_megatron, world_size=world_size)


def _check_compute_log_probs_without_megatron(rank: int, world_size: int, port: int) -> None:
    sys.modules["megatron"] = None  # importing any of it now raises, as on an install without it
    math_utils._MEGATRON_INSTALLED = False
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group([0])
        logits = _full_logits(torch.float32)[:, 0]
        tokens = _target()[:, 0]
        # [seq, 1], the shape compute_log_probs has always returned
        expected = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))

        torch.testing.assert_close(compute_log_probs(logits, tokens, group), expected, atol=1e-6, rtol=1e-6)
    finally:
        dist.destroy_process_group()


def test_compute_log_probs_runs_without_megatron() -> None:
    """The FSDP and torchtitan log-prob path imports nothing from Megatron when it is not installed."""
    run_multiprocess(_check_compute_log_probs_without_megatron, world_size=1)
