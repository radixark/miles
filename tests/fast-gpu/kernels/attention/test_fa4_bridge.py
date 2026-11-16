"""The trainer's FA4 attention is bitwise SGLang's FA4 decode, and its gradients are FA4's backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="needs SM100"
)

H, HKV, D = 32, 4, 128
SCALE = D**-0.5


def _inputs(lens, seed=0):
    torch.manual_seed(seed)
    total = sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, HKV, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v, cu


@pytest.mark.parametrize("lens", [[100], [129, 256, 300], [2048], [513, 3000]])
def test_trainer_forward_is_bitwise_sglang_decode(lens):
    from miles.backends.fsdp_utils.sglang_attn_bridge.hf_fa4_patch import FA4AttnFunction
    from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache

    q, k, v, cu = _inputs(lens)
    total = q.shape[0]
    with torch.no_grad():
        trained = FA4AttnFunction.apply(q, k, v, cu, max(lens), SCALE)
    doc = torch.repeat_interleave(torch.arange(len(lens), device="cuda"), torch.tensor(lens, device="cuda"))
    pos = torch.arange(total, device="cuda") - cu[:-1][doc]
    page_table = cu[:-1][doc][:, None] + torch.arange(max(lens), device="cuda", dtype=torch.int32)[None, :]
    page_table = torch.minimum(page_table, (cu[:-1][doc] + pos)[:, None]).to(torch.int32)
    decoded = flash_attn_with_kvcache(
        q=q,
        k_cache=k.view(total, 1, HKV, D),
        v_cache=v.view(total, 1, HKV, D),
        page_table=page_table,
        cache_seqlens=(pos + 1).to(torch.int32),
        cu_seqlens_q=torch.arange(total + 1, device="cuda", dtype=torch.int32),
        max_seqlen_q=1,
        softmax_scale=SCALE,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        num_splits=1,
    )
    assert torch.equal(trained, decoded)


def test_a_document_does_not_depend_on_what_it_is_packed_with():
    """The old-logprob and training passes may pack a document with different neighbours, padding
    and longest document; its attention output must not change."""
    from miles.backends.fsdp_utils.sglang_attn_bridge.hf_fa4_patch import FA4AttnFunction

    def run(lens, doc):
        q, k, v, cu = _inputs(lens, seed=3)
        s = sum(lens[:doc])
        torch.manual_seed(7)  # the same document contents in every packing
        qd, kd, vd = (torch.randn(lens[doc], *t.shape[1:], device="cuda", dtype=torch.bfloat16) for t in (q, k, v))
        q[s : s + lens[doc]], k[s : s + lens[doc]], v[s : s + lens[doc]] = qd, kd, vd
        with torch.no_grad():
            out = FA4AttnFunction.apply(q, k, v, cu, max(lens), SCALE)
        return out[s : s + lens[doc]]

    alone = run([600], 0)
    for lens, doc in [([300, 600], 1), ([600, 7000], 0), ([5, 600, 128, 1], 1), ([4096, 600, 3], 1)]:
        assert torch.equal(run(lens, doc), alone), (lens, doc)


def test_gradients_match_the_fp32_reference():
    from miles.backends.fsdp_utils.sglang_attn_bridge.hf_fa4_patch import FA4AttnFunction

    lens = [700, 33, 1500]
    q, k, v, cu = _inputs(lens, seed=1)
    dout = torch.randn_like(q)

    def grads(fn, dtype):
        leaves = [t.detach().to(dtype).requires_grad_() for t in (q, k, v)]
        fn(*leaves).backward(dout.to(dtype))
        return [t.grad.float() for t in leaves]

    def reference(q, k, v):
        outs = []
        for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
            qq, kk, vv = (t[s:e].transpose(0, 1) for t in (q, k, v))
            kk, vv = (t.repeat_interleave(H // HKV, 0) for t in (kk, vv))
            outs.append(torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=True, scale=SCALE))
        return torch.cat(outs, 1).transpose(0, 1)

    got = grads(lambda q, k, v: FA4AttnFunction.apply(q, k, v, cu, max(lens), SCALE), torch.bfloat16)
    want = grads(reference, torch.float32)
    for g, w in zip(got, want):
        assert (g - w).norm() / w.norm() < 1e-2
    again = grads(lambda q, k, v: FA4AttnFunction.apply(q, k, v, cu, max(lens), SCALE), torch.bfloat16)
    assert all(torch.equal(a, b) for a, b in zip(got, again)), "the backward must be deterministic"
