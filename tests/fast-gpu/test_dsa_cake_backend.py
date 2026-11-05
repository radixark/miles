"""``--dsa-attention-backend cake`` through the miles DSA dispatchers on SM100a / SM103a / SM107a, one GPU.

* ``sparse_attention(config=CAKE_SPARSE_ATTENTION_CONFIG)`` vs the default per-arch dispatch at the TP-local shapes of
  the four wired models -- GLM-5 / DeepSeek-V3.2 (thd, RoPE tail), DeepSeek-V4 (bshd batch, FP32 sink), GLM-5.3-Flash
  (thd NoPE, padded to 576 as its layer passes it) and DeepSeek-V4.1 (128 window + 512 compressed slots with ``-1``
  holes, sink) -- in both Cake variants; ``det`` is bit-identical across passes.
* The Cake indexer scores vs the current-main Triton scores (GLM-5 32 heads thd, DeepSeek-V4 64 heads sbhd): the same
  -inf entries, finite scores within SCORE_TOL (two FP32 sums in different orders), and the canonical top-k picks the
  same keys; and the SCORE_ROW_ALIGN padding of unaligned key counts.
"""

import sys

import pytest
import torch

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1200, suite="stage-c-4-gpu-b200", num_gpus=1, labels=["megatron"], hardware=["blackwell"])

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (10, 7)}:
    pytest.skip("the Cake DSA kernels target SM100a / SM103a / SM107a", allow_module_level=True)

import miles_ext_kernels  # noqa: E402
import miles_ext_kernels.cake.dsa as ext  # noqa: E402

from miles.kernels.attention.dsa import indexer_logits, indexer_logits_sbhd, sparse_attention  # noqa: E402
from miles.kernels.attention.dsa.indexer import CAKE_INDEXER_CONFIG, SCORE_ROW_ALIGN  # noqa: E402
from miles.kernels.attention.dsa.sparse_attention import CAKE_SPARSE_ATTENTION_CONFIG  # noqa: E402
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402

DEVICE = "cuda"
TOL = 2e-2  # max |error| / max |reference| between two BF16-output kernels with FP32 accumulation
# The Cake and the Triton indexer sum the FP32 head products in different orders, so their scores differ in the last
# bits: at most 1.6e-7 of the largest |score| on B200 and GB300, each within 2.6e-7 of the FP64 scores. The bound is
# about 6x that.
SCORE_TOL = 1e-6


@pytest.fixture(params=["det", "fast"])
def variant(request):
    miles_ext_kernels.configure(cake_dsa_variant=request.param)
    yield request.param
    miles_ext_kernels.configure()  # back to the defaults: det


def _rel_err(actual, expected):
    actual, expected = actual.float(), expected.float()
    scale = expected.abs().max().clamp_min(1e-6)
    return ((actual - expected).abs().max() / scale).item()


def _causal_indices(rows, topk, num_keys, gen):
    limits = ((torch.arange(rows, device=DEVICE) + 1) * num_keys // rows).clamp(min=1, max=num_keys)
    score = torch.rand(rows, num_keys, device=DEVICE, generator=gen)
    score = score.masked_fill(torch.arange(num_keys, device=DEVICE)[None, :] >= limits[:, None], -1.0)
    picked = torch.topk(score, min(topk, num_keys), dim=-1)
    idx = torch.where(picked.values >= 0, picked.indices, -1).int()
    return torch.nn.functional.pad(idx, (0, topk - idx.shape[1]), value=-1)


def _window_compressed_indices(seq, compressed_topk, gen, window=128, ratio=4):
    rows = torch.arange(seq, device=DEVICE)
    window_idx = rows[:, None] - (window - 1) + torch.arange(window, device=DEVICE)[None, :]
    window_idx = torch.where(window_idx >= 0, window_idx, -1)
    n_comp = seq // ratio
    score = torch.rand(seq, n_comp, device=DEVICE, generator=gen)
    score = score.masked_fill(torch.arange(n_comp, device=DEVICE)[None, :] >= ((rows + 1) // ratio)[:, None], -1.0)
    picked = torch.topk(score, compressed_topk, dim=-1)
    comp_idx = torch.where(picked.values >= 0, picked.indices + seq, -1)
    comp_idx = comp_idx[:, torch.randperm(compressed_topk, device=DEVICE, generator=gen)]
    return torch.cat([window_idx, comp_idx], dim=-1).int(), seq + n_comp


def _run(q, kv, indices, do, sm_scale, *, d_v=None, sink=None, config=None):
    q_ = q.clone().requires_grad_(True)
    kv_ = kv.clone().requires_grad_(True)
    s_ = None if sink is None else sink.clone().requires_grad_(True)
    out = sparse_attention(q_, kv_, indices, sm_scale, d_v=d_v, attn_sink=s_, config=config)
    out.backward(do)
    return [t for t in (out.detach(), q_.grad, kv_.grad, None if s_ is None else s_.grad) if t is not None]


def _glm5(gen, heads, rows, topk, d_qk=576, nope_pad=False):
    q = torch.randn(1, rows, heads, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(1, rows, 1, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    if nope_pad:  # GLM-5.3-Flash: a 512-wide NoPE latent padded with a zero tail to 576
        q[..., 512:] = 0
        kv[..., 512:] = 0
    indices = _causal_indices(rows, topk, rows, gen).view(1, rows, 1, topk)
    do = torch.randn(1, rows, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices, do


def _dsv4(gen, heads, batch, seq, indices, seq_kv):
    q = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(batch, seq_kv, 1, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    sink = torch.randn(heads, device=DEVICE, generator=gen)
    do = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices.unsqueeze(2), sink, do


def _cases():
    """(name, inputs builder) per wired model at its TP-local shape."""

    def glm5_h16(gen):
        q, kv, indices, do = _glm5(gen, 16, 1024, 512)
        return (q, kv, indices, do, 576**-0.5), {"d_v": 512}

    def glm53_flash_h64(gen):
        q, kv, indices, do = _glm5(gen, 64, 1024, 512, nope_pad=True)
        return (q, kv, indices, do, 256**-0.5), {"d_v": 512}

    def dsv4_sink_h16(gen):
        batch, seq = 2, 512
        indices = torch.stack([_causal_indices(seq, 192, seq, gen) for _ in range(batch)])
        q, kv, indices, sink, do = _dsv4(gen, 16, batch, seq, indices, seq)
        return (q, kv, indices, do, 512**-0.5), {"sink": sink}

    def dsv41_h16(gen):
        seq = 2048
        indices, seq_kv = _window_compressed_indices(seq, 512, gen)
        q, kv, indices, sink, do = _dsv4(gen, 16, 1, seq, indices.unsqueeze(0), seq_kv)
        return (q, kv, indices, do, 512**-0.5), {"sink": sink}

    return {
        "glm5-h16": glm5_h16,
        "glm5.3-flash-h64": glm53_flash_h64,
        "dsv4-sink-h16": dsv4_sink_h16,
        "dsv4.1-h16": dsv41_h16,
    }


@pytest.mark.parametrize("case", list(_cases()))
def test_cake_attention_matches_the_default_dispatch(variant, case):
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    args, kwargs = _cases()[case](gen)

    cake = _run(*args, config=CAKE_SPARSE_ATTENTION_CONFIG, **kwargs)
    default = _run(*args, **kwargs)
    for name, ours, theirs in zip(("out", "dq", "dkv", "dsink")[: len(cake)], cake, default, strict=True):
        err = _rel_err(ours, theirs)
        print(f"{case} {variant} {name}: cake vs default rel err {err:.3e}")
        assert err < TOL, name
    if variant == "det":
        again = _run(*args, config=CAKE_SPARSE_ATTENTION_CONFIG, **kwargs)
        assert all(torch.equal(a, b) for a, b in zip(cake, again, strict=True)), "det cake pass is not bit-identical"


def _assert_scores_match(cake, triton):
    """The same out-of-range (-inf) entries, every other score finite and within SCORE_TOL of the largest |score|."""
    masked = triton == float("-inf")
    assert torch.equal(cake == float("-inf"), masked), "the out-of-range (-inf) entries differ"
    assert torch.isfinite(cake[~masked]).all() and torch.isfinite(triton[~masked]).all()
    error = ((cake[~masked] - triton[~masked]).abs().max() / triton[~masked].abs().max()).item()
    assert error <= SCORE_TOL, f"max |diff| / max |score| {error:.3e}"


def _canonical_topk(logits, topk, starts, ends):
    """The canonical top-k of the score rows of ``logits``, on an aligned copy: its proposal kernel writes just left of
    each row's range."""
    rows = logits.reshape(-1, logits.shape[-1])
    num_keys = rows.shape[-1]
    padded = -(-num_keys // SCORE_ROW_ALIGN) * SCORE_ROW_ALIGN
    copy = torch.empty(rows.shape[0], padded, dtype=rows.dtype, device=rows.device)[:, :num_keys]
    copy.copy_(rows)
    return get_dsa_topk_fn("canonical")(copy, topk, starts, ends)


def test_cake_glm5_indexer_scores_match_the_triton_scores_and_pick_the_same_keys():
    gen = torch.Generator(device=DEVICE).manual_seed(2)
    rows, heads = 2048, 32
    q = torch.randn(rows, heads, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(rows, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    w = torch.randn(rows, heads, device=DEVICE, generator=gen)
    starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    ends = (torch.arange(rows, device=DEVICE) + 1).int()

    cake = indexer_logits(q, k, w, starts, ends, config=CAKE_INDEXER_CONFIG)
    triton = indexer_logits(q, k, w, starts, ends)
    assert cake.stride(0) % SCORE_ROW_ALIGN == 0
    _assert_scores_match(cake, triton)
    assert torch.equal(_canonical_topk(cake, 512, starts, ends), _canonical_topk(triton, 512, starts, ends))


def test_cake_dsv4_batched_indexer_scores_match_the_triton_scores_and_pick_the_same_keys():
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    seq, batch, heads, ratio = 2048, 2, 64, 4
    q = torch.randn(seq, batch, heads, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(seq // ratio, batch, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    w = torch.randn(seq, batch, heads, device=DEVICE, generator=gen)
    starts = torch.zeros(seq, device=DEVICE, dtype=torch.int32)
    ends = ((torch.arange(seq, device=DEVICE) + 1) // ratio).int()

    cake = indexer_logits_sbhd(q, k, w, starts, ends, config=CAKE_INDEXER_CONFIG)
    triton = indexer_logits_sbhd(q, k, w, starts, ends)
    assert cake.shape == (batch, seq, seq // ratio) and cake.stride(1) % SCORE_ROW_ALIGN == 0
    _assert_scores_match(cake, triton)
    picks = [_canonical_topk(scores, 128, starts.repeat(batch), ends.repeat(batch)) for scores in (cake, triton)]
    assert torch.equal(*picks)


@pytest.mark.parametrize("seq_len_kv", [512, 509], ids=["aligned", "padded"])
def test_cake_indexer_rows_match_the_ext_kernels_and_keep_the_row_alignment(seq_len_kv):
    gen = torch.Generator(device=DEVICE).manual_seed(4)
    q = torch.randn(512, 32, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(seq_len_kv, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    w = torch.randn(512, 32, device=DEVICE, generator=gen)
    starts = torch.zeros(512, device=DEVICE, dtype=torch.int32)
    ends = (torch.arange(512, device=DEVICE) + 1).clamp(max=seq_len_kv).int()

    logits = indexer_logits(q, k, w, starts, ends, config=CAKE_INDEXER_CONFIG)
    assert logits.shape == (512, seq_len_kv) and logits.stride(0) % SCORE_ROW_ALIGN == 0
    assert torch.equal(logits, ext.indexer_logits(q, k, w, starts, ends, layout="thd"))
    picked = _canonical_topk(logits, 64, starts, ends)
    expected = torch.topk(logits, 64, dim=-1)
    expected = torch.where(torch.isfinite(expected.values), expected.indices.int(), -1).sort(dim=-1).values
    assert torch.equal(picked.sort(dim=-1).values, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
