"""Tests for the DeepSeek-V4 compressor under THD packing and Context Parallel.

Compressing a packed stream must equal compressing each sample alone, and compacting it per CP
rank then mapping the gathered rows back must equal compressing it once.

Not in CI: the compressor imports the fp8 kernels, and its bf16-in/fp32-out matmul has no CPU
kernel outside a ROCm build. The CP collectives are slicing and concatenation here; the real
ones are checked by tests/e2e/precision/test_dsv4_thd_cp_correctness.py.
"""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.deepseek_v4.ops.thd_utils import (
    CompressorInputCompact,
    ThdLayout,
    compact_gather_index,
    compact_group_capacity,
    compressed_cu_seqlens,
    compressed_rank_layout,
    compressor_boundary_width,
)

try:
    from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
except ImportError:
    DeepSeekV4Compressor = None


def requires_fp8_kernels():
    return pytest.mark.skipif(DeepSeekV4Compressor is None, reason="fp8 kernels not installed")


def requires_rocm_build():
    return pytest.mark.skipif(
        torch.version.hip is None,
        reason="the compressor's bf16 matmul has no CPU kernel outside a ROCm build",
    )


HEAD_DIM = 128
HIDDEN = 256


def _thd(cu, *, max_seqlen=0, compressed_group_ids=None):
    """ThdLayout for a single-rank packed stream."""
    return ThdLayout(
        cu_seqlens=cu,
        global_start=0,
        max_seqlen=max_seqlen,
        compressed_group_ids=compressed_group_ids,
    )


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)


def _pad_to(lens, multiple):
    """Append a pad segment, the way data.py does, so the stream divides evenly."""
    rem = (multiple - sum(lens) % multiple) % multiple
    lens = list(lens) + [rem] if rem else list(lens)
    return lens, sum(lens)


def _compressor(compress_ratio):
    torch.manual_seed(0)
    c = DeepSeekV4Compressor(
        config=SimpleNamespace(
            hidden_size=HIDDEN,
            qk_pos_emb_head_dim=64,
            layernorm_epsilon=1e-6,
            fp8=None,
            dsv4_compress_rope_theta=160000,
            original_max_position_embeddings=65536,
            rotary_scaling_factor=4,
            beta_fast=32,
            beta_slow=1,
        ),
        head_dim=HEAD_DIM,
        compress_ratio=compress_ratio,
        rotate=False,
    )
    # A checkpoint fills these in production; left alone they are raw memory, i.e. NaN.
    with torch.no_grad():
        c.ape.normal_(0, 0.02)
        c.wkv.weight.normal_(0, 0.02)
        c.wgate.weight.normal_(0, 0.02)
    return c


def _packed_input(lens):
    torch.manual_seed(1)
    return torch.randn(sum(lens), 1, HIDDEN, dtype=torch.bfloat16)


def _per_segment_reference(compressor, x, lens):
    """Compress each segment alone through the BSHD path, then concatenate.

    Trims each segment to a whole number of groups first, mirroring the THD cutoff rule.
    """
    ratio = compressor.compress_ratio
    out, start = [], 0
    for n in lens:
        cutoff = (n // ratio) * ratio
        if cutoff:
            seg = x[start : start + cutoff].transpose(0, 1)
            out.append(compressor.forward_raw(seg).transpose(0, 1))
        start += n
    return torch.cat(out, dim=0) if out else None


def _cp_compress(compressor, x, cu, cp_size, max_seqlen):
    """Run the CP path without a process group.

    The boundary exchange is a slice of what the neighbour owns; the all-gather is a cat.
    """
    ratio = compressor.compress_ratio
    d_comp = compressor_boundary_width(ratio)
    l_local = x.size(0) // cp_size
    c_cap = compact_group_capacity(l_local, ratio)

    per_rank = []
    for cp_rank in range(cp_size):
        start = cp_rank * l_local
        boundary = x.new_zeros((d_comp,) + tuple(x.shape[1:]))
        if start >= d_comp:
            boundary = x[start - d_comp : start]
        elif start > 0:
            boundary[-start:] = x[:start]
        compact, comp_ids = CompressorInputCompact.apply(x[start : start + l_local], boundary, cu, start, ratio, c_cap)
        kv, cu_comp = compressor._forward_thd(compact, _thd(cu, max_seqlen=max_seqlen, compressed_group_ids=comp_ids))
        assert cu_comp is None  # the pre-grouped path cannot derive it
        per_rank.append(kv)

    mapping = compressed_rank_layout(
        cu, compressed_cu_seqlens(cu, ratio), l_local=l_local, cp_size=cp_size, ratio=ratio, c_cap=c_cap
    )
    rank_major = torch.cat(per_rank, dim=0)
    return rank_major.index_select(0, mapping.clamp(min=0).long()), mapping


PACKED_SHAPES = [
    pytest.param(4, [8, 8], id="r4-exact-multiples"),
    pytest.param(4, [10, 7], id="r4-both-with-a-tail"),
    pytest.param(4, [40, 24, 64], id="r4-three-segments"),
    pytest.param(128, [3 * 128 + 5, 2 * 128, 128 + 127], id="r128-mixed-tails"),
    pytest.param(128, [128, 300], id="r128-exact-then-tail"),
]


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("ratio,lens", PACKED_SHAPES)
def test_packed_matches_per_segment(ratio, lens):
    """The oracle is the BSHD path, a separate implementation, so this one needs a tolerance.
    The CP checks below compare two runs of the same path and must agree bit for bit.
    """
    compressor = _compressor(ratio)
    x = _packed_input(lens)
    packed, cu_comp = compressor._forward_thd(x, _thd(_cu(lens)))
    reference = _per_segment_reference(compressor, x, lens)
    assert cu_comp.tolist() == [0] + torch.tensor([n // ratio for n in lens]).cumsum(0).tolist()
    torch.testing.assert_close(packed.float(), reference.float(), rtol=2e-2, atol=2e-2)


CP_SHAPES = [
    pytest.param(4, [10, 22], id="r4-none-divisible"),
    pytest.param(4, [3, 29], id="r4-first-shorter-than-ratio"),
    pytest.param(4, [32], id="r4-single-segment-spans-all-ranks"),
    pytest.param(4, [4, 4, 4, 20], id="r4-more-segments-than-ranks"),
    pytest.param(4, [16, 16], id="r4-segment-edge-on-the-cp-split"),
    pytest.param(4, [1, 1, 1, 29], id="r4-several-segments-shorter-than-ratio"),
    pytest.param(4, [31, 1], id="r4-tail-segment-of-one-token"),
    pytest.param(128, [512, 256, 1280], id="r128-uniform"),
    pytest.param(128, [3, 1789], id="r128-first-shorter-than-ratio"),
    pytest.param(128, [1536, 512, 7], id="r128-short-tail"),
]


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("ratio,lens", CP_SHAPES)
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_cp_chain_matches_whole_stream(ratio, lens, cp_size):
    """Compacting per rank, compressing, then mapping the gathered rows back to sequence order
    must equal compressing the packed stream once.

    Covers the compaction, the fixed capacity, comp_ids driving RoPE and the overlap transform,
    and the rank-major row map. Trailing tokens get no compressed entry, so none may be invented.
    """
    lens, _ = _pad_to(lens, cp_size * ratio)
    cu = _cu(lens)
    compressor = _compressor(ratio)
    x = _packed_input(lens)
    reference, _ = compressor._forward_thd(x, _thd(cu))
    if reference is None:
        pytest.skip("no segment reaches the ratio, nothing to compare")
    got, _ = _cp_compress(compressor, x, cu, cp_size, max(lens))
    total_comp = int(compressed_cu_seqlens(cu, ratio)[-1])
    assert reference.shape[0] == total_comp
    torch.testing.assert_close(got[:total_comp], reference, rtol=0, atol=0)


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_chain_backward_matches_whole_stream(cp_size):
    """The compaction's scatter must send each source row's gradient back where it came from.

    Unlike the forward, this needs a tolerance: the scatter is an index_add_, and the two paths
    accumulate into a row in a different order.
    """
    ratio, lens = 4, [40, 24, 64]
    cu = _cu(lens)
    compressor = _compressor(ratio)
    base = _packed_input(lens).float()

    ref_x = base.clone().requires_grad_(True)
    compressor._forward_thd(ref_x, _thd(cu))[0].sum().backward()

    cp_x = base.clone().requires_grad_(True)
    _cp_compress(compressor, cp_x, cu, cp_size, max(lens))[0].sum().backward()

    torch.testing.assert_close(cp_x.grad, ref_x.grad, rtol=1e-5, atol=1e-6)


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_short_segments_produce_no_group(cp_size):
    """A segment shorter than `ratio` contributes nothing, and a rank holding only such
    segments emits zero real groups rather than a padded one.
    """
    ratio = 128
    lens, _ = _pad_to([3 * ratio, ratio - 1, 5], cp_size * ratio)
    cu = _cu(lens)
    cu_comp = compressed_cu_seqlens(cu, ratio)
    assert cu_comp.tolist()[:3] == [0, 3, 3]

    _, mapping = _cp_compress(_compressor(ratio), _packed_input(lens), cu, cp_size, max(lens))
    assert int((mapping >= 0).sum()) == int(cu_comp[-1])

    packed, cu_comp_tiny = _compressor(ratio)._forward_thd(_packed_input([5, 7]), _thd(_cu([5, 7])))
    assert packed is None
    assert cu_comp_tiny.tolist() == [0, 0, 0]


@requires_fp8_kernels()
@requires_rocm_build()
def test_pre_grouped_requires_max_seqlen():
    """Pre-grouped input needs max_seqlen: its group ids address positions past the compacted
    row count, so falling back to x.size(0) would index out of the rope table.
    """
    ratio, lens = 4, [40, 24]
    cu = _cu(lens)
    c_cap = compact_group_capacity(sum(lens), ratio)
    _, comp_ids = compact_gather_index(cu, global_start=0, l_local=sum(lens), ratio=ratio, c_cap=c_cap)
    compact = torch.zeros(c_cap * ratio, 1, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_seqlen"):
        _compressor(ratio)._forward_thd(compact, _thd(cu, max_seqlen=None, compressed_group_ids=comp_ids))
