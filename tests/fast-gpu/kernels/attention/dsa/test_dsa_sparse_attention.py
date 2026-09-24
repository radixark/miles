import importlib.util
import pathlib

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

tilelang = pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa import sparse_attention  # noqa: E402
from miles.kernels.attention.dsa.sparse_attention import flash_mla_sparse_fwd  # noqa: E402

BACKENDS = ["tilelang"] + (["flash_mla"] if flash_mla_sparse_fwd is not None else [])

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
    (1, 128, 8, 1, 512, 64, 160, 64, False),  # TP-local head count below FlashMLA's minimum
    (1, 128, 16, 1, 512, 64, 160, 64, False),
    (1, 256, 64, 1, 512, 64, 320, 128, False),
    (1, 256, 128, 1, 512, 64, 320, 2048, False),  # topk > seq_len_kv exercises -1 padding
]


def IDS(cases):
    return [
        f"b{c[0]}_s{c[1]}_h{c[2]}_g{c[3]}_dv{c[4]}_dt{c[5]}_kv{c[6]}_top{c[7]}_{'sink' if c[8] else 'nosink'}"
        for c in cases
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", MQA_CASES + MLA_CASES, ids=IDS(MQA_CASES + MLA_CASES))
def test_forward_matches_reference(case, backend):
    torch.manual_seed(0)
    batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink = case
    q, kv, indices, attn_sink = _inputs(batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink)
    sm_scale = (d_v + d_tail) ** -0.5
    ref = reference.sparse_attention_ref(q, kv, indices, sm_scale, d_v, attn_sink)
    out = sparse_attention(q, kv, indices, sm_scale, d_v=d_v, attn_sink=attn_sink, forward_backend=backend)
    assert out.shape == (batch, seq_len, heads, d_v)
    assert reference.rel_diff(ref, out) < 1e-3
    assert (ref - out.float()).abs().max() < 0.1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", MQA_CASES[:4] + MLA_CASES[:3], ids=IDS(MQA_CASES[:4] + MLA_CASES[:3]))
def test_backward_matches_autograd(case, backend):
    torch.manual_seed(0)
    batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink = case
    q, kv, indices, attn_sink = _inputs(batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink)
    sm_scale = (d_v + d_tail) ** -0.5

    grad_out = torch.randn(batch, seq_len, heads, d_v, device="cuda", dtype=torch.float32)

    q_ref, kv_ref = q.float().requires_grad_(), kv.float().requires_grad_()
    sink_ref = attn_sink.clone().requires_grad_() if sink else None
    (reference.sparse_attention_ref(q_ref, kv_ref, indices, sm_scale, d_v, sink_ref) * grad_out).sum().backward()

    q_tl, kv_tl = q.clone().requires_grad_(), kv.clone().requires_grad_()
    sink_tl = attn_sink.clone().requires_grad_() if sink else None
    out = sparse_attention(q_tl, kv_tl, indices, sm_scale, d_v=d_v, attn_sink=sink_tl, forward_backend=backend)
    (out.float() * grad_out).sum().backward()

    # bf16 GEMMs and atomic dKV accumulation, so the tolerance is looser than the forward.
    assert reference.rel_diff(q_ref.grad, q_tl.grad) < 0.05
    assert reference.rel_diff(kv_ref.grad, kv_tl.grad) < 0.05
    if sink:
        assert reference.rel_diff(sink_ref.grad, sink_tl.grad) < 0.05


@pytest.mark.skipif(len(BACKENDS) < 2, reason="FlashMLA not installed")
@pytest.mark.parametrize("case", MQA_CASES + MLA_CASES, ids=IDS(MQA_CASES + MLA_CASES))
def test_forward_backends_agree(case):
    """Both forwards feed the same TileLang backward, so the log-sum-exp conversion must be exact enough
    that gradients agree to bf16 noise."""
    torch.manual_seed(0)
    batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink = case
    q, kv, indices, attn_sink = _inputs(batch, seq_len, heads, groups, d_v, d_tail, seq_len_kv, topk, sink)
    sm_scale = (d_v + d_tail) ** -0.5
    grad_out = torch.randn(batch, seq_len, heads, d_v, device="cuda", dtype=torch.float32)
    grads = {}
    for backend in BACKENDS:
        q_, kv_ = q.clone().requires_grad_(), kv.clone().requires_grad_()
        sink_ = attn_sink.clone().requires_grad_() if sink else None
        out = sparse_attention(q_, kv_, indices, sm_scale, d_v=d_v, attn_sink=sink_, forward_backend=backend)
        (out.float() * grad_out).sum().backward()
        grads[backend] = (out, q_.grad, kv_.grad, sink_.grad if sink else None)
    for a, b in zip(grads["tilelang"], grads["flash_mla"], strict=True):
        if a is not None:
            assert reference.rel_diff(a, b) < 1e-5


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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sink", [False, True], ids=["nosink", "sink"])
def test_rows_with_no_selected_keys(backend, sink):
    """Rows whose indices are all -1 attend to nothing (or only the sink): zero output, finite gradients."""
    torch.manual_seed(0)
    batch, seq_len, heads, d_v, seq_len_kv, topk = 1, 128, 64, 512, 160, 64
    q, kv, indices, attn_sink = _inputs(batch, seq_len, heads, 1, d_v, 0, seq_len_kv, topk, sink)
    indices[:, :5] = -1
    sm_scale = d_v**-0.5
    grad_out = torch.randn(batch, seq_len, heads, d_v, device="cuda", dtype=torch.float32)

    q_ref, kv_ref = q.float().requires_grad_(), kv.float().requires_grad_()
    sink_ref = attn_sink.clone().requires_grad_() if sink else None
    ref = reference.sparse_attention_ref(q_ref, kv_ref, indices, sm_scale, d_v, sink_ref)
    (ref * grad_out).sum().backward()

    q_tl, kv_tl = q.clone().requires_grad_(), kv.clone().requires_grad_()
    sink_tl = attn_sink.clone().requires_grad_() if sink else None
    out = sparse_attention(q_tl, kv_tl, indices, sm_scale, d_v=d_v, attn_sink=sink_tl, forward_backend=backend)
    (out.float() * grad_out).sum().backward()

    assert torch.all(out[:, :5] == 0)
    for actual in (out, q_tl.grad, kv_tl.grad) + ((sink_tl.grad,) if sink else ()):
        assert torch.isfinite(actual).all()
    assert reference.rel_diff(ref, out) < 1e-3
    assert reference.rel_diff(q_ref.grad, q_tl.grad) < 0.05
    assert reference.rel_diff(kv_ref.grad, kv_tl.grad) < 0.05
    if sink:
        assert reference.rel_diff(sink_ref.grad, sink_tl.grad) < 0.05
