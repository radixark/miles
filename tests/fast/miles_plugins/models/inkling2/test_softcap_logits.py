"""Inkling2's in-place logit softcap and padded-vocab mask: forward bit-exact against the unfused
fp32 graph, backward against autograd, fp32 logits overwritten, and only the output kept."""

import pytest
import torch

pytest.importorskip("mbridge")
from miles_plugins.models.inkling2 import ops  # noqa: E402

CAP = 30.0


def _reference(logits, cap, padded):
    out = cap * torch.tanh(logits.float() / cap)
    return out.masked_fill(padded, torch.finfo(out.dtype).min)


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Several backward chunks plus a ragged tail.
    monkeypatch.setattr(ops, "SOFTCAP_LOGITS_CHUNK_ROWS", 3)


@pytest.mark.parametrize("shape", [(1, 1, 11), (7, 1, 11), (10, 2, 16)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matches_unfused_graph(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(shape[0])
    leaf = (40 * torch.randn(*shape)).to(dtype).requires_grad_()
    leaf_ref = leaf.detach().clone().requires_grad_()
    padded = torch.arange(shape[-1]) >= shape[-1] - 3

    ref = _reference(leaf_ref * 1, CAP, padded)
    logits = leaf * 1  # the LM head's output, never a leaf
    out = ops.softcap_masked_logits(logits, CAP, padded)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    if dtype == torch.float32:
        assert out.data_ptr() == logits.data_ptr(), "fp32 logits must be overwritten, not copied"
    saved = out.grad_fn.saved_tensors
    assert len(saved) == 2 and saved[0].data_ptr() == out.data_ptr(), "backward must keep only the output"

    # The loss reads probabilities, so the padded columns' upstream gradient is finite but ignored.
    grad = torch.randn_like(ref)
    out.backward(grad.clone())
    ref.backward(grad)
    assert leaf.grad.dtype == dtype
    assert not leaf.grad[..., padded].any(), "padded columns must get no gradient"
    # Backward reads tanh back as out / cap instead of the saved tanh: within one ulp in bf16, and
    # within 1e-6 absolute in fp32 (measured worst over 600 seeds: 4.8e-7).
    tol = dict(rtol=2**-7, atol=0) if dtype == torch.bfloat16 else dict(rtol=0, atol=1e-6)
    torch.testing.assert_close(leaf.grad, leaf_ref.grad, **tol)


def test_no_grad_forward_matches() -> None:
    logits = 40 * torch.randn(5, 1, 13)
    padded = torch.arange(13) >= 10
    expected = _reference(logits, CAP, padded)
    with torch.no_grad():
        torch.testing.assert_close(ops.softcap_masked_logits(logits.clone(), CAP, padded), expected, rtol=0, atol=0)
