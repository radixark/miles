import importlib.util
import pathlib

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

tilelang = pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa import sparse_attention  # noqa: E402

_spec = importlib.util.spec_from_file_location("dsa_reference", pathlib.Path(__file__).with_name("dsa_reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def _random_indices(batch, seq_len, groups, seq_len_kv, topk):
    actual = min(topk, seq_len_kv)
    rows = [torch.randperm(seq_len_kv, device="cuda")[:actual] for _ in range(batch * seq_len * groups)]
    idx = torch.stack(rows).view(batch, seq_len, groups, actual).int()
    if topk > actual:
        idx = torch.nn.functional.pad(idx, (0, topk - actual), value=-1)
    return idx


def _inputs(batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink):
    q = torch.randn(batch, seq_len, heads, d_v + d_tail, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len_kv, groups, d_v + d_tail, device="cuda", dtype=torch.bfloat16)
    indices = _random_indices(batch, seq_len, groups, seq_len_kv, topk)
    attn_sink = torch.randn(heads, device="cuda", dtype=torch.float32) if sink else None
    return q, kv, indices, attn_sink


# (batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink)
MQA_CASES = [  # DeepSeek-V4: single latent, attention sink
    (1, 128, 8, 1, 512, 0, 160, 64, True),
    (2, 128, 8, 1, 512, 0, 160, 64, True),
    (1, 256, 64, 1, 512, 0, 320, 128, True),
    (1, 256, 8, 1, 512, 0, 320, 100, True),  # topk not a multiple of the kernel block
    (1, 128, 8, 1, 512, 0, 160, 64, False),
]
MLA_CASES = [  # GLM-5 / DeepSeek-V3.2: RoPE tail, no sink
    (1, 128, 16, 1, 512, 64, 160, 64, False),
    (1, 256, 64, 1, 512, 64, 320, 128, False),
    (1, 256, 128, 1, 512, 64, 320, 2048, False),  # topk > seq_len_kv exercises -1 padding
]


def IDS(cases):
    return [
        f"b{c[0]}_s{c[1]}_h{c[2]}_g{c[3]}_dv{c[4]}_dt{c[5]}_kv{c[6]}_top{c[7]}_{'sink' if c[8] else 'nosink'}"
        for c in cases
    ]


@pytest.mark.parametrize("case", MQA_CASES + MLA_CASES, ids=IDS(MQA_CASES + MLA_CASES))
def test_forward_matches_reference(case):
    torch.manual_seed(0)
    batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink = case
    q, kv, indices, attn_sink = _inputs(batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink)
    sm_scale = (d_v + d_tail) ** -0.5
    ref = reference.sparse_attention_ref(q, kv, indices, sm_scale, d_v, attn_sink)
    out = sparse_attention(q, kv, indices, sm_scale, d_v=d_v, attn_sink=attn_sink)
    assert out.shape == (batch, seq_len, heads, d_v)
    assert reference.rel_diff(ref, out) < 1e-3
    assert (ref - out.float()).abs().max() < 0.1


@pytest.mark.parametrize("case", MQA_CASES[:3] + MLA_CASES[:2], ids=IDS(MQA_CASES[:3] + MLA_CASES[:2]))
def test_backward_matches_autograd(case):
    torch.manual_seed(0)
    batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink = case
    q, kv, indices, attn_sink = _inputs(batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink)
    sm_scale = (d_v + d_tail) ** -0.5

    q_ref, kv_ref = q.float().requires_grad_(), kv.float().requires_grad_()
    sink_ref = attn_sink.clone().requires_grad_() if sink else None
    reference.sparse_attention_ref(q_ref, kv_ref, indices, sm_scale, d_v, sink_ref).sum().backward()

    q_tl, kv_tl = q.clone().requires_grad_(), kv.clone().requires_grad_()
    sink_tl = attn_sink.clone().requires_grad_() if sink else None
    sparse_attention(q_tl, kv_tl, indices, sm_scale, d_v=d_v, attn_sink=sink_tl).float().sum().backward()

    # bf16 GEMMs and atomic dKV accumulation, so the tolerance is looser than the forward.
    assert reference.rel_diff(q_ref.grad, q_tl.grad) < 0.05
    assert reference.rel_diff(kv_ref.grad, kv_tl.grad) < 0.05
    if sink:
        assert reference.rel_diff(sink_ref.grad, sink_tl.grad) < 0.05


def test_sink_changes_output():
    torch.manual_seed(0)
    q, kv, indices, _ = _inputs(1, 128, 8, 1, 512, 0, 160, 64, sink=False)
    sm_scale = 512**-0.5
    out_zero = sparse_attention(q, kv, indices, sm_scale, attn_sink=torch.zeros(8, device="cuda"))
    out_large = sparse_attention(q, kv, indices, sm_scale, attn_sink=torch.full((8,), 10.0, device="cuda"))
    assert (out_zero.float() - out_large.float()).abs().max() > 1e-3


def test_query_with_no_valid_key_returns_zero():
    torch.manual_seed(0)
    q, kv, indices, attn_sink = _inputs(1, 64, 8, 1, 512, 0, 80, 64, sink=True)
    indices[0, 0] = -1
    out = sparse_attention(q, kv, indices, 512**-0.5, attn_sink=attn_sink)
    assert torch.all(torch.isfinite(out))
    assert torch.all(out[0, 0] == 0)
