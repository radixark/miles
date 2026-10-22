"""The plain-torch vocab-parallel cross entropy must compute what Megatron's fused kernel does."""

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
        local_before = local.detach().clone()

        loss = vocab_parallel_cross_entropy(local, _target(), group)
        loss.backward(_grad_output())

        torch.testing.assert_close(loss, expected_loss, atol=tol, rtol=tol)
        assert local.grad.dtype == dtype
        torch.testing.assert_close(
            local.grad, expected_grad[..., rank * shard : (rank + 1) * shard], atol=tol, rtol=tol
        )
        torch.testing.assert_close(local.detach(), local_before, atol=0, rtol=0)


def _check_against_megatron(rank: int, world_size: int, group: dist.ProcessGroup) -> None:
    """Where Megatron is installed, the stand-in computes what its fused kernel computes."""
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

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


def _check_compute_log_probs_without_megatron(group: dist.ProcessGroup) -> None:
    """The FSDP and torchtitan log-prob path imports nothing from Megatron when it is not installed."""
    sys.modules["megatron"] = None  # importing any of it now raises, as on an install without it
    math_utils._MEGATRON_INSTALLED = False
    logits = _full_logits(torch.float32)[:, 0]
    tokens = _target()[:, 0]
    # compute_log_probs returns [seq, 1]
    expected = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))

    torch.testing.assert_close(compute_log_probs(logits, tokens, group), expected, atol=1e-6, rtol=1e-6)
    # None: the default group
    torch.testing.assert_close(compute_log_probs(logits, tokens, None), expected, atol=1e-6, rtol=1e-6)


def _run_checks(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(list(range(world_size)))
        _check_against_full_softmax(rank, world_size, group)
        _check_against_megatron(rank, world_size, group)
        if world_size == 1:
            # last: it hides Megatron from the rest of this process
            _check_compute_log_probs_without_megatron(group)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_vocab_parallel_cross_entropy(world_size: int) -> None:
    """Loss and gradient match a full-vocabulary log-softmax and Megatron's kernel on every TP shard."""
    run_multiprocess(_run_checks, world_size=world_size)


def _find_spec_seeing(found: dict[str, bool]):
    def find_spec(name: str):
        if name.split(".")[0] not in found:  # importing the parent package fails, as find_spec does
            raise ModuleNotFoundError(name)
        return object() if found.get(name, False) else None

    return find_spec


@pytest.mark.parametrize(
    "found,installed",
    [
        ({"megatron": True, "megatron.core": True}, True),
        ({"megatron": True}, False),  # megatron-energon's bare namespace, no megatron.core
        ({}, False),  # no megatron package at all
    ],
)
def test_megatron_counts_as_installed_only_with_megatron_core(monkeypatch, found, installed) -> None:
    monkeypatch.setattr(math_utils.importlib.util, "find_spec", _find_spec_seeing(found))
    assert math_utils._megatron_core_installed() is installed
