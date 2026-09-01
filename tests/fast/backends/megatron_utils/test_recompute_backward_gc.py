"""The checkpointed-backward collector is opt-in, idempotent, and does not change gradients."""

import gc

import pytest
import torch

from miles.backends.megatron_utils import recompute_gc


@pytest.fixture(autouse=True)
def _uninstalled(monkeypatch):
    """Each test starts from "not installed" and leaves megatron.core as it found it."""
    from megatron.core.tensor_parallel.random import CheckpointFunction

    original = CheckpointFunction.backward
    monkeypatch.setattr(recompute_gc, "_installed", False, raising=False)
    yield
    CheckpointFunction.backward = staticmethod(original)


def test_disabled_unless_the_environment_asks(monkeypatch):
    monkeypatch.delenv("MILES_RECOMPUTE_BACKWARD_GC_GEN", raising=False)
    from megatron.core.tensor_parallel.random import CheckpointFunction

    before = CheckpointFunction.backward
    assert recompute_gc.enable_recompute_backward_gc() is False
    assert CheckpointFunction.backward is before


def test_installs_once_and_collects_once_per_backward(monkeypatch):
    monkeypatch.setenv("MILES_RECOMPUTE_BACKWARD_GC_GEN", "2")
    from megatron.core.tensor_parallel.random import CheckpointFunction, checkpoint

    before = CheckpointFunction.backward
    assert recompute_gc.enable_recompute_backward_gc() is True
    installed = CheckpointFunction.backward
    assert installed is not before

    # Idempotent: a second call must not wrap the wrapper.
    assert recompute_gc.enable_recompute_backward_gc() is True
    assert CheckpointFunction.backward is installed

    generations = []
    real_collect = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *a: generations.append(a) or real_collect(*a))

    x = torch.randn(4, 4, requires_grad=True)
    checkpoint(lambda t: (t * 2).sin(), False, x).sum().backward()

    assert generations == [(2,)]


def test_gradients_are_unchanged(monkeypatch):
    monkeypatch.setenv("MILES_RECOMPUTE_BACKWARD_GC_GEN", "2")
    from megatron.core.tensor_parallel.random import checkpoint

    recompute_gc.enable_recompute_backward_gc()

    x = torch.randn(8, 8, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)

    checkpoint(lambda t: (t * 2).sin(), False, x).sum().backward()
    (reference * 2).sin().sum().backward()

    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("value", ["-1", "not-a-number"])
def test_bad_or_negative_values_disable_it(monkeypatch, value):
    monkeypatch.setenv("MILES_RECOMPUTE_BACKWARD_GC_GEN", value)
    assert recompute_gc.recompute_backward_gc_generation() == -1
    assert recompute_gc.enable_recompute_backward_gc() is False
