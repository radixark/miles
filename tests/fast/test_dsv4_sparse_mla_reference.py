"""CPU tests for the DeepSeek-V4 sparse-MQA reference backward and how it gets selected.

The reference exists for GPUs whose shared memory cannot host the fused backward kernel. Both
things it has to get right are cheap to check without a GPU:

  * gathering the top-k KV rows must not make autograd allocate a gradient the size of the
    *unexpanded* KV cross product, and
  * it must not be chosen for a shape whose backward would overflow a 32-bit element count.
"""

import pytest
import torch

from miles_plugins.models.deepseek_v4.ops.kernel.torch_sparse_mla import sparse_attn_torch


def _reference_inputs(B=2, S=7, H=4, D=16, S_kv=11, topk=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=g, dtype=torch.float32)
    kv = torch.randn(B, S_kv, D, generator=g, dtype=torch.float32)
    attn_sink = torch.randn(H, generator=g, dtype=torch.float32)
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), generator=g)
    # Mask a few slots so the -1 path is covered.
    topk_idxs[0, 0, 0] = -1
    topk_idxs[-1, -1, -1] = -1
    return q, kv, attn_sink, topk_idxs


def _expanded_gather_reference(q, kv, attn_sink, topk_idxs, sm_scale):
    """The straightforward writing: gather from a [B, S, S_kv, D] expand of kv."""
    B, S, H, D = q.shape
    valid = topk_idxs != -1
    safe_idx = topk_idxs.masked_fill(~valid, 0).long()
    kv_gathered = torch.gather(
        kv.unsqueeze(1).expand(B, S, kv.shape[1], D),
        2,
        safe_idx.unsqueeze(-1).expand(-1, -1, -1, D),
    )
    scores = torch.einsum("bshd,bstd->bsht", q.float(), kv_gathered.float()) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, H, 1).expand(B, S, H, 1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bsht,bstd->bshd", probs, kv_gathered.float()).to(q.dtype)


def test_flat_gather_matches_the_expanded_writing_forward_and_backward():
    """Same function, so the forward and dq/dsink agree bit for bit.

    dkv does not, and cannot: the expanded writing scatters into [B, S, S_kv, D] and then reduces
    along S, while the flat one scatter-adds straight into [B, S_kv, D]. Same terms, different
    summation order, and the reference computes in fp32 throughout (it casts its inputs), so the
    two disagree at fp32 rounding.
    """
    q, kv, attn_sink, topk_idxs = _reference_inputs()
    sm_scale = q.shape[-1] ** -0.5

    inputs = [(t.clone().requires_grad_(True), t.clone().requires_grad_(True)) for t in (q, kv, attn_sink)]
    (q_a, q_b), (kv_a, kv_b), (sink_a, sink_b) = inputs

    out_a = sparse_attn_torch(q_a, kv_a, sink_a, topk_idxs, sm_scale)
    out_b = _expanded_gather_reference(q_b, kv_b, sink_b, topk_idxs, sm_scale)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)

    grad = torch.randn_like(out_a)
    out_a.backward(grad)
    out_b.backward(grad)
    torch.testing.assert_close(q_a.grad, q_b.grad, rtol=0, atol=0, msg="q gradient differs")
    torch.testing.assert_close(sink_a.grad, sink_b.grad, rtol=0, atol=0, msg="attn_sink gradient differs")
    torch.testing.assert_close(kv_a.grad, kv_b.grad, rtol=1e-6, atol=1e-6, msg="kv gradient differs")


def test_masked_slots_do_not_contribute():
    """A -1 index is a masked slot: its KV row must receive no gradient through that slot."""
    q, kv, attn_sink, topk_idxs = _reference_inputs(B=1, S=1, H=2, D=8, S_kv=6, topk=3)
    topk_idxs[:] = torch.tensor([[[-1, -1, -1]]])
    kv = kv.double().requires_grad_(True)
    out = sparse_attn_torch(q.double(), kv, attn_sink.double(), topk_idxs, sm_scale=0.125)

    # Every slot is masked, so the softmax keeps only the sink and the output is exactly zero.
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    out.sum().backward()
    torch.testing.assert_close(kv.grad, torch.zeros_like(kv.grad), rtol=0, atol=0)


def test_kv_gradient_is_shaped_like_kv_not_like_the_cross_product():
    """The bug this guards against allocated [B, S, S_kv, D] -- 2048 GiB at production shapes."""
    q, kv, attn_sink, topk_idxs = _reference_inputs()
    kv = kv.requires_grad_(True)
    sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale=0.25).sum().backward()
    assert kv.grad.shape == kv.shape


def test_reference_backward_is_refused_past_int_max():
    tilelang_sparse_mla = pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla",
        reason="needs tilelang",
    )
    representable = tilelang_sparse_mla._reference_backward_is_representable

    small = torch.empty(1, 128, 8, 64, device="meta")
    assert representable(small, torch.empty(1, 128, 32, device="meta"))

    # A 128K DeepSeek-V4 layer under CP=8: 1 * 16384 * 640 * 512 = 5.37e9 elements.
    big = torch.empty(1, 16384, 64, 512, device="meta")
    assert not representable(big, torch.empty(1, 16384, 640, device="meta"))
