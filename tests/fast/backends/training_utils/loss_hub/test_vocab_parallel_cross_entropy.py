"""The vocab-parallel cross entropy behind every backend's torch log-probs."""

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.loss.hub.math_utils import compute_log_probs, vocab_parallel_cross_entropy

SEQ, BATCH, VOCAB = 5, 2, 12


def _full_logits(dtype: torch.dtype) -> torch.Tensor:
    return (torch.randn(SEQ, BATCH, VOCAB, generator=torch.Generator().manual_seed(7)) * 4).to(dtype)


def _target() -> torch.Tensor:
    # each TP=2 shard's first and last id
    return torch.tensor([[0, 11], [5, 6], [6, 5], [11, 0], [3, 9]])


def _reference(full: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = full.detach().clone().requires_grad_()
    loss = -torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss.backward(_grad_output())
    return loss.detach(), logits.grad


def _grad_output() -> torch.Tensor:
    return torch.randn(SEQ, BATCH, generator=torch.Generator().manual_seed(11))


def _check_against_full_softmax(rank: int, world_size: int, group: dist.ProcessGroup) -> None:
    """Loss and gradient equal a plain log-softmax over the whole vocabulary, however it is split."""
    shard = VOCAB // world_size
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 1e-2)):
        full = _full_logits(dtype)
        expected_loss, expected_grad = _reference(full, _target())
        local = full[..., rank * shard : (rank + 1) * shard].clone().requires_grad_()

        loss = vocab_parallel_cross_entropy(local, _target(), group)
        loss.backward(_grad_output())

        torch.testing.assert_close(loss, expected_loss, atol=tol, rtol=tol)
        assert local.grad.dtype == dtype
        torch.testing.assert_close(
            local.grad, expected_grad[..., rank * shard : (rank + 1) * shard], atol=tol, rtol=tol
        )


def _check_against_megatron(rank: int, world_size: int, group: dist.ProcessGroup) -> None:
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy as megatron_cross_entropy

    def megatron(logits, target, group):
        return megatron_cross_entropy(logits, target, tp_group=group)

    shard = VOCAB // world_size
    for dtype in (torch.float32, torch.bfloat16):
        full = _full_logits(dtype)
        results = []
        for fn in (vocab_parallel_cross_entropy, megatron):
            local = full[..., rank * shard : (rank + 1) * shard].clone().requires_grad_()
            loss = fn(local, _target(), group)
            loss.backward(_grad_output())
            results.append((loss.detach(), local.grad))
        (loss, grad), (megatron_loss, megatron_grad) = results
        assert torch.equal(loss, megatron_loss) and torch.equal(grad, megatron_grad), dtype


def _check_compute_log_probs(group: dist.ProcessGroup) -> None:
    logits = _full_logits(torch.float32)[:, 0]
    tokens = _target()[:, 0]
    # compute_log_probs returns [seq, 1]
    expected = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))

    torch.testing.assert_close(compute_log_probs(logits.clone(), tokens, group), expected, atol=1e-6, rtol=1e-6)


def _run_checks(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(list(range(world_size)))
        _check_against_full_softmax(rank, world_size, group)
        if world_size == 1:
            _check_compute_log_probs(group)
    finally:
        dist.destroy_process_group()


def _run_megatron_check(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _check_against_megatron(rank, world_size, dist.new_group(list(range(world_size))))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_vocab_parallel_cross_entropy(world_size: int) -> None:
    """Loss and gradient equal a plain log-softmax over the whole vocabulary, however it is split."""
    run_multiprocess(_run_checks, world_size=world_size)


@pytest.mark.parametrize("world_size", [1, 2])
def test_matches_megatron_vocab_parallel_cross_entropy(world_size: int) -> None:
    """Bit for bit Megatron's math. Its fused kernel differs in the last bits, from torch.compile's
    exp and summation order, and rounds the gradient to bf16."""
    pytest.importorskip("megatron.core.tensor_parallel.cross_entropy")
    run_multiprocess(_run_megatron_check, world_size=world_size)
