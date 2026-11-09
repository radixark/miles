import functools

import pytest
import torch
import torch.nn.functional as F

from miles.backends.torchtitan_utils.document_attention import document_ids
from miles.kernels.tpu.gated_delta import document_gated_delta, document_short_conv

# documents shorter than, as long as and longer than the 4-wide kernel, one token long, and a padding tail
_DOC_LENGTHS = [1, 3, 4, 9, 2, 5]


def _positions(doc_lengths: list[int]) -> torch.Tensor:
    return torch.cat([torch.arange(n) for n in doc_lengths])


def _per_document_conv(x_TC: torch.Tensor, weight_C1W: torch.Tensor, doc_lengths: list[int]) -> torch.Tensor:
    outs = []
    for x_doc_TC in x_TC.split(doc_lengths):
        x_1CT = F.pad(x_doc_TC.t().unsqueeze(0), (weight_C1W.shape[-1] - 1, 0))
        outs.append(F.conv1d(x_1CT, weight_C1W, groups=weight_C1W.shape[0]).squeeze(0).t())
    return F.silu(torch.cat(outs))


@pytest.mark.parametrize("doc_lengths", [_DOC_LENGTHS, [2]], ids=["packed", "shorter_than_kernel"])
def test_short_conv_matches_conv1d_run_on_each_document(doc_lengths):
    torch.manual_seed(0)
    x = torch.randn(sum(doc_lengths), 6, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(6, 1, 4, dtype=torch.float32, requires_grad=True)

    out = document_short_conv(x, weight, document_ids(_positions(doc_lengths)))
    expected = _per_document_conv(x, weight, doc_lengths)
    torch.testing.assert_close(out, expected)

    grad = torch.randn_like(out)
    torch.testing.assert_close(
        torch.autograd.grad(out, (x, weight), grad), torch.autograd.grad(expected, (x, weight), grad)
    )


def test_short_conv_keeps_the_input_dtype():
    x = torch.randn(8, 4, dtype=torch.bfloat16)
    out = document_short_conv(x, torch.randn(4, 1, 4, dtype=torch.bfloat16), document_ids(torch.arange(8)))
    assert out.dtype == torch.bfloat16


def _recurrent_gated_delta(q_THK, k_THK, v_THV, g_TH, beta_TH):
    """The delta rule one token at a time (torchtitan's GDN test reference), on one document."""
    q_THK = q_THK * torch.rsqrt(q_THK.pow(2).sum(-1, keepdim=True) + 1e-6) * q_THK.shape[-1] ** -0.5
    k_THK = k_THK * torch.rsqrt(k_THK.pow(2).sum(-1, keepdim=True) + 1e-6)
    state_HKV = q_THK.new_zeros(v_THV.shape[1], q_THK.shape[-1], v_THV.shape[-1])
    outs = []
    for q_HK, k_HK, v_HV, g_H, beta_H in zip(q_THK, k_THK, v_THV, g_TH, beta_TH, strict=True):
        state_HKV = state_HKV * g_H.exp()[:, None, None]
        delta_HV = (v_HV - torch.einsum("hkv,hk->hv", state_HKV, k_HK)) * beta_H[:, None]
        state_HKV = state_HKV + torch.einsum("hk,hv->hkv", k_HK, delta_HV)
        outs.append(torch.einsum("hkv,hk->hv", state_HKV, q_HK))
    return torch.stack(outs)


def _per_document_gated_delta(q_THK, k_THK, v_THV, g_TH, beta_TH, doc_lengths):
    group = v_THV.shape[1] // q_THK.shape[1]
    q_THK, k_THK = q_THK.repeat_interleave(group, dim=1), k_THK.repeat_interleave(group, dim=1)
    pieces = zip(*(t.split(doc_lengths) for t in (q_THK, k_THK, v_THV, g_TH, beta_TH)), strict=True)
    return torch.cat([_recurrent_gated_delta(*piece) for piece in pieces])


# with 8-token chunks: documents that end on a chunk boundary, span three chunks, or fit inside one
_CHUNKED_DOC_LENGTHS = [1, 7, 8, 13, 3, 20, 5]


@pytest.mark.parametrize("chunk_size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("num_key_heads", [4, 2], ids=["mha", "grouped_values"])
def test_chunked_delta_rule_matches_the_recurrence_run_on_each_document(num_key_heads, chunk_size):
    torch.manual_seed(0)
    seq_len = sum(_CHUNKED_DOC_LENGTHS)
    q, k = (torch.randn(seq_len, num_key_heads, 16, requires_grad=True) for _ in range(2))
    v = torch.randn(seq_len, 4, 8, requires_grad=True)
    g = (-F.softplus(torch.randn(seq_len, 4))).requires_grad_()
    beta = torch.rand(seq_len, 4, requires_grad=True)
    inputs = (q, k, v, g, beta)

    out = document_gated_delta(*inputs, document_ids(_positions(_CHUNKED_DOC_LENGTHS)), chunk_size=chunk_size)
    expected = _per_document_gated_delta(*inputs, _CHUNKED_DOC_LENGTHS)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)

    grad = torch.randn_like(out)
    got, want = torch.autograd.grad(out, inputs, grad), torch.autograd.grad(expected, inputs, grad)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_chunk128_matches_recurrence_across_multiple_chunks():
    torch.manual_seed(0)
    doc_lengths = [1, 127, 128, 129, 3, 260]
    seq_len = sum(doc_lengths)
    q, k = (torch.randn(seq_len, 2, 16, requires_grad=True) for _ in range(2))
    v = torch.randn(seq_len, 4, 8, requires_grad=True)
    g = (-F.softplus(torch.randn(seq_len, 4))).requires_grad_()
    beta = torch.rand(seq_len, 4, requires_grad=True)
    inputs = (q, k, v, g, beta)

    out = document_gated_delta(*inputs, document_ids(_positions(doc_lengths)), chunk_size=128)
    expected = _per_document_gated_delta(*inputs, doc_lengths)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)

    grad = torch.randn_like(out)
    got, want = torch.autograd.grad(out, inputs, grad), torch.autograd.grad(expected, inputs, grad)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("decay", [-0.01, -150.0], ids=["weak_decay", "strong_decay"])
def test_chunked_delta_rule_matches_recurrence_at_decay_limits(decay):
    torch.manual_seed(0)
    seq_len = sum(_CHUNKED_DOC_LENGTHS)
    q, k = (torch.randn(seq_len, 2, 16, requires_grad=True) for _ in range(2))
    v = torch.randn(seq_len, 4, 8, requires_grad=True)
    g = torch.full((seq_len, 4), decay, requires_grad=True)
    beta = torch.rand(seq_len, 4, requires_grad=True)
    inputs = (q, k, v, g, beta)

    out = document_gated_delta(*inputs, document_ids(_positions(_CHUNKED_DOC_LENGTHS)), chunk_size=16)
    expected = _per_document_gated_delta(*inputs, _CHUNKED_DOC_LENGTHS)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)

    grad = torch.randn_like(out)
    got, want = torch.autograd.grad(out, inputs, grad), torch.autograd.grad(expected, inputs, grad)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_chunked_delta_rule_runs_under_selective_activation_checkpointing(compiled):
    """Selective checkpointing handles scan's dispatch and never asks AOT to cut through a differentiated scan."""
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

    torch.manual_seed(0)
    seq_len = sum(_CHUNKED_DOC_LENGTHS)
    q, k = (torch.randn(seq_len, 2, 16, requires_grad=True) for _ in range(2))
    v = torch.randn(seq_len, 4, 8, requires_grad=True)
    g = (-F.softplus(torch.randn(seq_len, 4))).requires_grad_()
    beta = torch.rand(seq_len, 4, requires_grad=True)
    inputs = (q, k, v, g, beta)
    ids = document_ids(_positions(_CHUNKED_DOC_LENGTHS))

    def run(*xs):
        return document_gated_delta(*xs, ids, chunk_size=16)

    expected = run(*inputs)

    def recompute_everything(ctx, op, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    contexts = functools.partial(create_selective_checkpoint_contexts, recompute_everything)

    def checkpointed(*xs):
        return checkpoint(run, *xs, use_reentrant=False, context_fn=contexts)

    out = (torch.compile(checkpointed, backend="aot_eager", fullgraph=True) if compiled else checkpointed)(*inputs)
    torch.testing.assert_close(out, expected)

    grad = torch.randn_like(out)
    got, want = torch.autograd.grad(out, inputs, grad), torch.autograd.grad(expected, inputs, grad)
    torch.testing.assert_close(got, want)
