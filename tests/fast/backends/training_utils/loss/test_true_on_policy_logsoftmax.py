import sys
import types

import torch

from miles.backends.training_utils.loss_hub.math_utils import (
    calculate_log_probs_and_entropy,
)


def test_top_logsoftmax_keeps_pytorch_default(monkeypatch):
    calls = []
    original = torch.log_softmax

    def tracked_log_softmax(*args, **kwargs):
        calls.append("torch")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "log_softmax", tracked_log_softmax)
    logits = torch.tensor([[2.0, 1.0, -1.0]])
    tokens = torch.tensor([1])

    log_probs, _ = calculate_log_probs_and_entropy(
        logits,
        tokens,
        None,
        true_on_policy=True,
    )

    assert calls == ["torch"]
    torch.testing.assert_close(
        log_probs,
        original(logits, dim=-1)[:, 1],
    )


def test_top_logsoftmax_uses_batch_invariant_kernel_only_when_requested(
    monkeypatch,
):
    calls = []
    original = torch.log_softmax
    module_name = "sglang.srt.batch_invariant_ops.batch_invariant_ops"
    fake_module = types.ModuleType(module_name)

    def batch_invariant_log_softmax(*args, **kwargs):
        calls.append("batch_invariant")
        return original(*args, **kwargs)

    fake_module.log_softmax = batch_invariant_log_softmax
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    logits = torch.tensor([[2.0, 1.0, -1.0]])
    tokens = torch.tensor([1])

    log_probs, _ = calculate_log_probs_and_entropy(
        logits,
        tokens,
        None,
        true_on_policy=True,
        batch_invariant=True,
    )

    assert calls == ["batch_invariant"]
    torch.testing.assert_close(
        log_probs,
        original(logits, dim=-1)[:, 1],
    )
