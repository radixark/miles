"""The Qwen3.8-Flash-Next TOP bindings (hyper-connection, PLE, QSA selection and attention): serving's programs
forward, the FP32 chain backward.

Qwen3.8-Flash-Next shapes. Forward: a row's bytes do not depend on which rows share its call (serving decodes it in a
small batch, the trainer scores it in a packed stream), a packed document is the document alone, and the trainer
binding is serving's program. Backward: each binding's gradients against FP32 autograd of the same math written
apart from the kernels.
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

# est_time is an upper bound until a /rerun-test run measures it.
register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from sglang.srt.true_on_policy import hyper_connection as sgl_hc  # noqa: E402
from sglang.srt.true_on_policy import qsa as sgl_qsa  # noqa: E402

from miles_plugins.top.qwen38_hc import top_hc_combine, top_hc_mix_inject  # noqa: E402
from miles_plugins.top.qwen38_ple import _TopPLEIncrement  # noqa: E402
from miles_plugins.top.qwen38_qsa import top_qsa_attention  # noqa: E402

STREAMS, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDE = STREAMS * HIDDEN
# Each side of the native program's small-batch dispatch thresholds (24/25 and 32/33 rows), one row, and a packed stream.
ROW_COUNTS = (1, 2, 7, 24, 25, 32, 33, 128, 1000, 4096)
# The FP32 chain differentiates at BF16 inputs and ignores the forward's BF16 rounding points. Measured on H200:
# every gradient of the three bindings is 0.20% to 0.27% from the reference.
BACKWARD_RTOL = 1e-2


def _generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _bf16(g, *shape, scale=1.0):
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _documents(lengths):
    """(first row, length) of each packed document."""
    return zip(itertools.accumulate(lengths[:-1], initial=0), lengths, strict=True)


def _leaf(t, dtype):
    return t.detach().to(dtype).clone().requires_grad_()


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def _assert_backward(got: dict, reference: dict):
    errors = {name: _rel(got[name], reference[name]) for name in reference}
    assert max(errors.values()) < BACKWARD_RTOL, errors


def _hc_weights(g):
    return dict(
        norm=_bf16(g, WIDE, scale=0.1),
        down=_bf16(g, LOWRANK, WIDE, scale=WIDE**-0.5),
        up=_bf16(g, WIDE, LOWRANK, scale=LOWRANK**-0.5),
        inject=_bf16(g, STREAMS, WIDE, scale=WIDE**-0.5),
    )


def _serving_hc(x, y, w):
    """Serving's gated residual around a sublayer output y: (the sublayer input, the write gates, the streams out)."""
    mixed, normed, _ = sgl_hc.hc_mix(x, norm_weight=w["norm"], w_down=w["down"], w_up=w["up"], n=STREAMS, eps=EPS)
    h_post = sgl_hc.hc_inject(normed, w_inject=w["inject"], n=STREAMS)
    return mixed, h_post, sgl_hc.hc_combine(x, y2d=y, h_post=h_post, n=STREAMS)


def _hc_reference(x, y, w):
    streams = x.view(-1, STREAMS, HIDDEN)
    normed = streams * torch.rsqrt(streams.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w["norm"].view(STREAMS, HIDDEN))
    normed = normed.view(-1, WIDE)
    gate = torch.sigmoid(F.linear(F.silu(F.linear(normed, w["down"]) / STREAMS), w["up"]))
    mixed = (gate * normed).view(-1, STREAMS, HIDDEN).mean(1)
    h_post = 2 * torch.sigmoid(F.linear(normed, w["inject"]) / STREAMS)
    return mixed, (streams + h_post.unsqueeze(-1) * y.unsqueeze(1)).view(-1, WIDE)


def test_a_hyper_connection_row_does_not_depend_on_its_batch():
    g = _generator(0)
    w = _hc_weights(g)
    x, y = _bf16(g, ROW_COUNTS[-1], WIDE), _bf16(g, ROW_COUNTS[-1], HIDDEN)
    whole = _serving_hc(x, y, w)
    for count in ROW_COUNTS:
        prefix = torch.arange(count, device="cuda")
        for rows in (prefix, prefix * (x.shape[0] // count)):
            got = _serving_hc(x[rows].contiguous(), y[rows].contiguous(), w)
            for name, a, b in zip(("mixed", "write gates", "streams"), got, whole, strict=True):
                assert torch.equal(a, b[rows]), f"{count} rows: {name}"


def test_the_trainer_hyper_connection_is_servings():
    """Megatron hands the streams over as [sequence, batch, 4 * hidden]."""
    g = _generator(1)
    w = _hc_weights(g)
    x, y = _bf16(g, 128, WIDE), _bf16(g, 128, HIDDEN)
    mixed, h_post, out = _serving_hc(x, y, w)
    got_mixed, got_post = top_hc_mix_inject(
        x.view(64, 2, WIDE), w["norm"], w["down"], w["up"], w["inject"], STREAMS, EPS
    )
    got_out = top_hc_combine(x.view(64, 2, WIDE), y.view(64, 2, HIDDEN), got_post, STREAMS)
    assert torch.equal(got_mixed.reshape(-1, HIDDEN), mixed)
    assert torch.equal(got_post.reshape(-1, STREAMS), h_post)
    assert torch.equal(got_out.reshape(-1, WIDE), out)


def test_the_hyper_connection_gradient_is_the_fp32_chains():
    g = _generator(2)
    w = _hc_weights(g)
    x, y = _bf16(g, 512, WIDE), _bf16(g, 512, HIDDEN)
    d_mixed = torch.randn(512, HIDDEN, generator=g, device="cuda")
    d_out = torch.randn(512, WIDE, generator=g, device="cuda")

    def grads(dtype, run):
        leaves = {name: _leaf(t, dtype) for name, t in dict(w, x=x, y=y).items()}
        mixed, out = run(leaves)
        torch.autograd.backward([mixed.float(), out.float()], [d_mixed, d_out])
        return {name: leaf.grad for name, leaf in leaves.items()}

    def top(v):
        mixed, h_post = top_hc_mix_inject(v["x"], v["norm"], v["down"], v["up"], v["inject"], STREAMS, EPS)
        return mixed, top_hc_combine(v["x"], v["y"], h_post, STREAMS)

    _assert_backward(grads(torch.bfloat16, top), grads(torch.float32, lambda v: _hc_reference(v["x"], v["y"], v)))


PLE_KERNEL, PLE_DILATION = 4, 3
PLE_NAMES = ("hc_state", "key", "value", "norm_key", "norm_query", "norm_conv", "conv1d_weight")


def _ple_inputs(g, rows):
    streams, key, value = _bf16(g, rows, WIDE), _bf16(g, rows, WIDE), _bf16(g, rows, HIDDEN)
    norms = [_bf16(g, WIDE, scale=0.1) for _ in range(3)]
    return [streams, key, value, *norms, _bf16(g, WIDE, 1, PLE_KERNEL, scale=0.5)]


def _ple_reference(streams, key, value, wk, wq, wc, conv, cu):
    """The PLE increment: the n-gram value gated per stream by its key's agreement with the stream, plus a causal
    dilated depthwise convolution of the normed gated value that starts over at every document."""
    rows = streams.shape[0]
    shape = (rows, STREAMS, HIDDEN)

    def normed(x, weight):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * (1 + weight.view(STREAMS, HIDDEN))

    score = (normed(key.view(shape), wk) * normed(streams.view(shape), wq)).sum(-1) / math.sqrt(HIDDEN)
    gate = torch.sigmoid(torch.sign(score) * torch.sqrt(score.abs().clamp_min(1e-6)))
    gated = gate[..., None] * value[:, None, :]
    taps_in = normed(gated, wc).reshape(rows, WIDE)
    position = torch.arange(rows, device=streams.device)
    start = cu.long()[torch.searchsorted(cu[1:].long(), position, right=True)]
    acc = torch.zeros_like(taps_in)
    for tap, weight in enumerate(conv.reshape(WIDE, PLE_KERNEL).unbind(1)):
        source = position - (PLE_KERNEL - 1 - tap) * PLE_DILATION
        inside = (source >= start).float()[:, None]
        acc = acc + weight * taps_in.index_select(0, source.clamp_min(0)) * inside
    return gated.reshape(rows, WIDE) + acc * torch.sigmoid(acc)


def test_a_packed_ple_document_is_the_document_alone():
    """Serving prefills a request from an empty convolution state; the trainer scores it inside a packed stream."""
    lengths = (5, 64, 300, 1)
    inputs = _ple_inputs(_generator(3), sum(lengths))
    packed = _TopPLEIncrement.apply(*inputs, STREAMS, EPS, PLE_DILATION, _cu(lengths))
    for lo, n in _documents(lengths):
        rows = [t[lo : lo + n].contiguous() for t in inputs[:3]]
        alone = _TopPLEIncrement.apply(*rows, *inputs[3:], STREAMS, EPS, PLE_DILATION, _cu([n]))
        assert torch.equal(packed[lo : lo + n], alone), f"document at row {lo}"


def test_the_ple_gradient_is_the_fp32_chains():
    g = _generator(4)
    lengths = (5, 64, 300, 1)
    inputs = _ple_inputs(g, sum(lengths))
    cu = _cu(lengths)
    d_out = torch.randn(sum(lengths), WIDE, generator=g, device="cuda")

    def grads(dtype, run):
        leaves = [_leaf(t, dtype) for t in inputs]
        run(leaves).float().backward(d_out)
        return {name: leaf.grad for name, leaf in zip(PLE_NAMES, leaves, strict=True)}

    top = grads(torch.bfloat16, lambda v: _TopPLEIncrement.apply(*v, STREAMS, EPS, PLE_DILATION, cu))
    _assert_backward(top, grads(torch.float32, lambda v: _ple_reference(*v, cu)))


QUERY_HEADS, KV_HEADS, HEAD_DIM, SLOTS = 24, 2, 256, 96
QSA_LENGTHS = (40, 300, 7)


def _qsa_inputs(g):
    """Packed documents; a row attends its own document's rows up to itself, a random SLOTS of them (ascending) once
    there are more."""
    rows = sum(QSA_LENGTHS)
    q = _bf16(g, rows, QUERY_HEADS, HEAD_DIM, scale=0.5)
    k = _bf16(g, rows, KV_HEADS, HEAD_DIM, scale=0.5)
    v = _bf16(g, rows, KV_HEADS, HEAD_DIM)
    slots = torch.full((rows, SLOTS), -1, dtype=torch.int32, device="cuda")
    for lo, n in _documents(QSA_LENGTHS):
        for t in range(n):
            visible = torch.arange(lo, lo + t + 1, device="cuda")
            if visible.numel() > SLOTS:
                keep = torch.randperm(visible.numel(), generator=g, device="cuda")[:SLOTS]
                visible = visible[keep].sort().values
            slots[lo + t, : visible.numel()] = visible.int()
    return q, k, v, slots


def _dense_attention(q, k, v, slots, scale):
    """Masked dense attention over the same keys: no gather of the selected slots."""
    rows = q.shape[0]
    allowed = torch.zeros(rows, rows, dtype=torch.bool, device=q.device)
    row = torch.arange(rows, device=q.device)[:, None].expand_as(slots)
    allowed[row[slots >= 0], slots[slots >= 0].long()] = True
    group = QUERY_HEADS // KV_HEADS
    scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(group, 1)) * scale
    weights = scores.masked_fill(~allowed[None], float("-inf")).softmax(-1)
    return torch.einsum("hts,shd->thd", weights, v.repeat_interleave(group, 1)).reshape(rows, -1)


def test_a_qsa_attention_row_does_not_depend_on_its_batch():
    """Decode attends one query row a request over the pool's keys; the trainer scores every row of the stream."""
    q, k, v, slots = _qsa_inputs(_generator(5))
    scale = HEAD_DIM**-0.5
    whole = top_qsa_attention(q, k, v, slots, scale)
    for rows in (torch.tensor([0]), torch.tensor([39, 40, 339, 346]), torch.arange(0, q.shape[0], 3)):
        rows = rows.cuda()
        got, _, _ = sgl_qsa.sparse_attention(q[rows].contiguous(), k, v, slots[rows].contiguous(), scale, aux=False)
        assert torch.equal(got.reshape(len(rows), -1), whole[rows]), rows.tolist()


def test_the_qsa_attention_gradient_is_the_fp32_chains():
    g = _generator(6)
    q, k, v, slots = _qsa_inputs(g)
    scale = HEAD_DIM**-0.5
    d_out = torch.randn(q.shape[0], QUERY_HEADS * HEAD_DIM, generator=g, device="cuda")

    def grads(dtype, run):
        leaves = [_leaf(t, dtype) for t in (q, k, v)]
        run(*leaves).float().backward(d_out)
        return {name: leaf.grad for name, leaf in zip(("query", "key", "value"), leaves, strict=True)}

    top = grads(torch.bfloat16, lambda *leaves: top_qsa_attention(*leaves, slots, scale))
    _assert_backward(top, grads(torch.float32, lambda *leaves: _dense_attention(*leaves, slots, scale)))


def test_qsa_block_selection_without_batch_invariant_mode_takes_tied_blocks_lowest_id_first():
    """The trainer does not set SGLang's batch-invariant flag; the shared block selection still takes tied scores lowest
    block id first, ids ascending and -1 padded, as serving does. Rows cover no block, at and around top-k, and many
    blocks."""
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode

    topk = 512
    counts = torch.tensor([0, 1, 511, 512, 513, 3000, 9000], dtype=torch.int32, device="cuda")
    # Four distinct non-negative scores: most blocks tie.
    scores = torch.randint(0, 4, (counts.numel(), int(counts.max())), generator=_generator(7), device="cuda") / 4
    with set_batch_invariant_mode(False):
        got = sgl_qsa.select_blocks(scores.float(), counts, topk)
    assert torch.equal(got.cpu(), sgl_qsa.select_blocks_reference(scores.float().cpu(), counts.cpu(), topk))
