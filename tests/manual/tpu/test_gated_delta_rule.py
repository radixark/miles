"""Compiled gated delta rule and short convolution against their CPU references on a real TPU.

Run on a TPU host: python -m pytest tests/manual/tpu/test_gated_delta_rule.py
"""

import functools

import pytest
import torch
import torch.nn.functional as F

torch_tpu = pytest.importorskip("torch_tpu")

from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts  # noqa: E402

from miles.kernels.tpu.gated_delta_rule import (  # noqa: E402
    allow_scan_in_selective_checkpoint,
    gated_delta_rule,
    short_conv,
)

_DOC_LENGTHS = [1, 7, 8, 13, 3, 20, 5]


def _segment_ids() -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(_DOC_LENGTHS), dtype=torch.int32), torch.tensor(_DOC_LENGTHS))


def _inputs():
    torch.manual_seed(0)
    seq_len = sum(_DOC_LENGTHS)
    q, k = (torch.randn(seq_len, 2, 16, requires_grad=True) for _ in range(2))
    v = torch.randn(seq_len, 4, 8, requires_grad=True)
    g = (-F.softplus(torch.randn(seq_len, 4))).requires_grad_()
    beta = torch.rand(seq_len, 4, requires_grad=True)
    return q, k, v, g, beta


@pytest.mark.parametrize("chunk_size", [16, 128])
def test_compiled_selective_checkpoint_matches_cpu(chunk_size):
    """Chunk 16 regresses TorchTPU's reverse scan; chunk 128 is the production default."""
    allow_scan_in_selective_checkpoint()
    torch.set_float32_matmul_precision("highest")
    inputs, ids = _inputs(), _segment_ids()
    expected = gated_delta_rule(*inputs, segment_ids=ids, chunk_size=chunk_size)
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, inputs, grad)

    tpu_ids = ids.to("tpu")

    def kernel(*xs):
        return gated_delta_rule(*xs, segment_ids=tpu_ids, chunk_size=chunk_size)

    def recompute_everything(ctx, op, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    contexts = functools.partial(create_selective_checkpoint_contexts, recompute_everything)

    def run(*xs):
        return checkpoint(kernel, *xs, use_reentrant=False, context_fn=contexts)

    tpu_inputs = tuple(t.detach().to("tpu").requires_grad_() for t in inputs)
    compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
    actual = compiled(*tpu_inputs)
    actual_grads = torch.autograd.grad(actual, tpu_inputs, grad.to("tpu"))

    torch.testing.assert_close(actual.cpu(), expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        tuple(t.cpu() for t in actual_grads), tuple(t.cpu() for t in expected_grads), atol=2e-5, rtol=2e-4
    )


def test_compiled_short_conv_matches_cpu():
    torch.manual_seed(0)
    seq_len = sum(_DOC_LENGTHS)
    x = torch.randn(seq_len, 6, requires_grad=True)
    weight = torch.randn(6, 1, 4, requires_grad=True)
    ids = _segment_ids()
    expected = short_conv(x, weight, segment_ids=ids)
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (x, weight), grad)

    tpu_x = x.detach().to("tpu").requires_grad_()
    tpu_weight = weight.detach().to("tpu").requires_grad_()
    compiled = torch.compile(short_conv, backend="tpu", fullgraph=True, dynamic=False)
    actual = compiled(tpu_x, tpu_weight, segment_ids=ids.to("tpu"))
    actual_grads = torch.autograd.grad(actual, (tpu_x, tpu_weight), grad.to("tpu"))

    torch.testing.assert_close(actual.cpu(), expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        tuple(t.cpu() for t in actual_grads), tuple(t.cpu() for t in expected_grads), atol=3e-3, rtol=3e-3
    )
