"""Inkling2's row-chunked capped SwiGLU: forward bit-exact against the unfused fp32 graph,
backward against autograd, and only the inputs kept for backward."""

import pytest
import torch

pytest.importorskip("mbridge")
from miles_plugins.models.inkling2 import ops  # noqa: E402

CAPS = [(0.0, 0.0), (7.0, 0.0), (0.0, 5.0), (7.0, 5.0)]


def _reference(fc1_out, probs, gate_cap, up_cap):
    gate, up = torch.chunk(fc1_out.float(), 2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    if gate_cap > 0:
        gate = gate_cap * torch.tanh(gate / gate_cap)
    if up_cap > 0:
        up = up_cap * torch.tanh(up / up_cap)
    return (gate * sigmoid * up * probs.float()).to(fc1_out.dtype)


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Several chunks plus a ragged tail.
    monkeypatch.setattr(ops, "CAPPED_SWIGLU_CHUNK_ROWS", 7)


@pytest.mark.parametrize("rows", [1, 7, 30])
@pytest.mark.parametrize("caps", CAPS)
def test_matches_unfused_graph(rows: int, caps: tuple[float, float]) -> None:
    torch.manual_seed(rows)
    x = (4 * torch.randn(rows, 16)).to(torch.bfloat16).requires_grad_()
    p = torch.rand(rows, 1).requires_grad_()
    x_ref, p_ref = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()

    out = ops.capped_swiglu(x, p, *caps)
    ref = _reference(x_ref, p_ref, *caps)
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert len(out.grad_fn.saved_tensors) == 2, "backward must keep only fc1_out and probs"

    grad = torch.randn_like(ref)
    out.backward(grad)
    ref.backward(grad)
    # Both round the fp32 gradient to bf16 once; the hand-written formula may land one ulp away.
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-4)


def test_skips_input_gradients_that_are_not_needed() -> None:
    x = torch.randn(9, 16, dtype=torch.bfloat16, requires_grad=True)
    p = torch.rand(9, 1)
    ops.capped_swiglu(x, p, 7.0, 5.0).sum().backward()
    assert x.grad is not None and p.grad is None


def test_other_layouts_use_the_unfused_graph() -> None:
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    p = torch.rand(2, 5, 1)
    torch.testing.assert_close(ops.capped_swiglu(x, p, 7.0, 5.0), _reference(x, p, 7.0, 5.0), rtol=0, atol=0)
