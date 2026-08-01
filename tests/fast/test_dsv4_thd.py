"""CPU unit tests for DeepSeek-V4's THD (packed varlen) path.

Both halves assert the same property: packing samples into one stream, and splitting that
stream across CP ranks, must give each sample exactly what it would get on its own.

The sparse-attention kernel takes no `cu_seqlens`, so `topk_idxs` alone decides what a query
can see; the compressor has to derive per-segment group positions for the same reason.

CP collectives are slicing and concatenation here. That the real ones behave that way is
checked on GPUs by tests/e2e/precision/test_dsv4_thd_cp_correctness.py.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
from miles_plugins.models.deepseek_v4.ops.thd_utils import (
    CompressorInputCompact,
    compact_gather_index,
    compact_group_capacity,
    compressed_cu_seqlens,
    compressed_rank_layout,
    compressor_boundary_width,
    get_compress_topk_idxs_thd,
    get_window_topk_idxs_thd,
    to_rank_major_rows,
)

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

WINDOW = 128
HEAD_DIM = 128
HIDDEN = 256

SHAPES = {
    "uniform": [512, 512, 512, 512],
    "short_tail": [1536, 512, 7],  # trailing pad segment, shorter than the longest
    "below_ratio": [1024, 3, 1021],  # a segment with no compressed group at ratio=128
    "exact_ratio": [128, 128, 1792],
    "single": [2048],
    "one_token": [2047, 1],
}


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)


def _starts(lens):
    return [0] + torch.tensor(lens).cumsum(0).tolist()


def _pad_to(lens, multiple):
    """Append a pad segment, the way data.py does, so the stream divides evenly."""
    rem = (multiple - sum(lens) % multiple) % multiple
    lens = list(lens) + [rem] if rem else list(lens)
    return lens, sum(lens)


# ======================================================================================
# index layer
# ======================================================================================


def _probe_tokens(lens):
    """Stride sweep, plus the tokens either side of every segment boundary."""
    total = sum(lens)
    probes = set(range(0, total, 37))
    for s in _starts(lens):
        probes.update(t for t in range(s - 2, s + 3) if 0 <= t < total)
    return sorted(probes)


def _reference_rows(lens, ratio, token):
    """KV rows `token` may read, derived one sample at a time."""
    starts = _starts(lens)
    seg = max(i for i, s in enumerate(starts[:-1]) if s <= token)
    seg_start, seg_len = starts[seg], lens[seg]
    pos = token - seg_start
    window = set(range(max(seg_start, token - WINDOW + 1), token + 1))
    if ratio == 0:
        return window, set()
    comp_start = sum(n // ratio for n in lens[:seg])
    n_visible = min((pos + 1) // ratio, seg_len // ratio)
    return window, set(range(comp_start, comp_start + n_visible))


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_packed_indices_match_running_each_sample_alone(ratio, shape):
    lens = SHAPES[shape]
    total = sum(lens)
    cu = _cu(lens)
    window = get_window_topk_idxs_thd(cu, window_size=WINDOW, total_tokens=total)[0]

    if ratio:
        cu_comp = compressed_cu_seqlens(cu, ratio)
        compress = get_compress_topk_idxs_thd(
            cu, cu_comp, ratio=ratio, total_tokens=total, max_n_compressed=max(1, max(lens) // ratio)
        )[0]

    for token in _probe_tokens(lens):
        want_window, want_compress = _reference_rows(lens, ratio, token)
        got_window = {int(i) for i in window[token] if i >= 0}
        assert got_window == want_window, f"{shape} ratio={ratio} token={token}"
        if ratio:
            got_compress = {int(i) - total for i in compress[token] if i >= 0}
            assert got_compress == want_compress, f"{shape} ratio={ratio} token={token}"


def _check_cp_slices(lens, total, ratio, cp_size, ctx):
    """Each rank's index slice must dereference to the entries the whole stream would give."""
    l_local = total // cp_size
    cu = _cu(lens)
    cu_comp = compressed_cu_seqlens(cu, ratio)
    max_n = max(1, max(lens) // ratio)
    c_cap = compact_group_capacity(l_local, ratio)
    mapping = compressed_rank_layout(cu, cu_comp, l_local=l_local, cp_size=cp_size, ratio=ratio, c_cap=c_cap)
    whole = get_compress_topk_idxs_thd(cu, cu_comp, ratio=ratio, total_tokens=total, max_n_compressed=max_n)

    for cp_rank in range(cp_size):
        start = cp_rank * l_local
        local = get_compress_topk_idxs_thd(
            cu,
            cu_comp,
            ratio=ratio,
            total_tokens=l_local,
            max_n_compressed=max_n,
            kv_offset=total,
            global_start=start,
            seq_to_rank_row=mapping,
        )
        want = whole[:, start : start + l_local]
        where = f"{ctx} rank={cp_rank}"
        assert torch.equal(local < 0, want < 0), where
        live = want >= 0
        assert torch.equal(local[live] - total, mapping[(want[live] - total).long()].long()), where


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_cp_slices_dereference_to_the_same_compressed_entries(ratio, cp_size, shape):
    lens, total = _pad_to(SHAPES[shape], cp_size)
    _check_cp_slices(lens, total, ratio, cp_size, f"{shape} ratio={ratio} cp={cp_size}")


def _draw(seed):
    """Random segments plus a pad segment, the shape data.py emits."""
    rng = random.Random(seed)
    ratio = rng.choice([4, 128])
    cp_size = rng.choice([1, 2, 4])
    lens = [rng.randint(1, 6 * ratio) for _ in range(rng.randint(1, 6))]
    # A rank must hold at least the boundary window.
    lens, total = _pad_to(lens, cp_size * max(compressor_boundary_width(ratio), 8))
    return ratio, cp_size, lens, total


@pytest.mark.parametrize("seed", range(20))
def test_random_streams_slice_and_dereference_consistently(seed):
    ratio, cp_size, lens, total = _draw(seed)
    _check_cp_slices(lens, total, ratio, cp_size, f"seed={seed} ratio={ratio} cp={cp_size} {lens}")


def _group_owners(lens, ratio, l_local):
    """Rank holding each compressed group's last token, in sequence-major order."""
    starts = _starts(lens)
    return [(starts[seg] + (c + 1) * ratio - 1) // l_local for seg, ln in enumerate(lens) for c in range(ln // ratio)]


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_each_compressed_group_lands_in_exactly_one_all_gather_slot(ratio, cp_size, shape):
    """A group belongs to the rank holding its last token; no row is dropped or aliased.

    A rank may also build the boundary group it does not own. That copy lands in a slot the map
    never points at, so it costs a slot but is never read.
    """
    lens, total = _pad_to(SHAPES[shape], cp_size)
    l_local = total // cp_size
    cu = _cu(lens)
    cu_comp = compressed_cu_seqlens(cu, ratio)
    n_total = int(cu_comp[-1])
    c_cap = compact_group_capacity(l_local, ratio)
    mapping = compressed_rank_layout(cu, cu_comp, l_local=l_local, cp_size=cp_size, ratio=ratio, c_cap=c_cap)

    real = mapping[:n_total]
    assert (real >= 0).all()
    assert (real < cp_size * c_cap).all()
    assert real.unique().numel() == n_total
    assert (mapping[n_total:] == -1).all()

    owners = _group_owners(lens, ratio, l_local)
    assert torch.div(real, c_cap, rounding_mode="floor").tolist() == owners

    buffer_len = l_local + compressor_boundary_width(ratio)
    for cp_rank in range(cp_size):
        gather, comp_ids = compact_gather_index(
            cu, global_start=cp_rank * l_local, l_local=l_local, ratio=ratio, c_cap=c_cap
        )
        assert int(gather.max()) < buffer_len
        assert owners.count(cp_rank) <= int((comp_ids >= 0).sum()) <= c_cap


def test_compaction_gradient_lands_on_exactly_the_source_rows():
    """Give one compact row a unit gradient; only the row it gathered may receive it.

    The scatter is an index_add_: a wrong index, or a missing keep mask, piles every padding
    slot onto boundary row 0. The forward looks fine; training is poisoned.
    """
    ratio, lens = 4, [10, 22]
    cu = _cu(lens)
    l_local, d_comp = 16, compressor_boundary_width(ratio)
    c_cap = compact_group_capacity(l_local, ratio)
    global_start = 16
    gather, _ = compact_gather_index(cu, global_start=global_start, l_local=l_local, ratio=ratio, c_cap=c_cap)
    for probe in range(c_cap * ratio):
        src = int(gather[probe])
        hidden = torch.zeros(l_local, 2, dtype=torch.float64, requires_grad=True)
        boundary = torch.zeros(d_comp, 2, dtype=torch.float64, requires_grad=True)
        compact, _ = CompressorInputCompact.apply(hidden, boundary, cu, global_start, ratio, c_cap)
        seed = torch.zeros_like(compact)
        seed[probe] = 1.0
        compact.backward(seed)
        grads = torch.cat([boundary.grad, hidden.grad], dim=0)
        if src < 0:
            assert torch.count_nonzero(grads) == 0, f"padding slot {probe} leaked gradient"
        else:
            assert torch.equal(grads[src], torch.ones(2, dtype=torch.float64))
            assert torch.count_nonzero(grads) == 2, f"slot {probe} touched more than its source"


def test_an_empty_compressed_stream_yields_no_rows():
    """A stream shorter than the ratio has no compressed rows, so the map has none to index."""
    rows, valid = to_rank_major_rows(
        torch.tensor([[0, 1, 7]], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert (rows == -1).all()
    assert not valid.any()


# ======================================================================================
# compressor
# ======================================================================================


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
        kv, cu_comp = compressor._forward_thd(compact, cu, compressed_group_ids=comp_ids, max_seqlen=max_seqlen)
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


@pytest.mark.parametrize("ratio,lens", PACKED_SHAPES)
def test_packed_compression_matches_compressing_each_segment_alone(ratio, lens):
    """The oracle is the BSHD path, a separate implementation, so this one needs a tolerance.
    The CP checks below compare two runs of the same path and must agree bit for bit.
    """
    compressor = _compressor(ratio)
    x = _packed_input(lens)
    packed, cu_comp = compressor._forward_thd(x, _cu(lens))
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


@pytest.mark.parametrize("ratio,lens", CP_SHAPES)
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_cp_chain_reproduces_the_whole_stream_compression(ratio, lens, cp_size):
    """Compacting per rank, compressing, then mapping the gathered rows back to sequence order
    must equal compressing the packed stream once.

    Covers the compaction, the fixed capacity, comp_ids driving RoPE and the overlap transform,
    and the rank-major row map. Trailing tokens get no compressed entry, so no rank may invent
    one for them.
    """
    lens, _ = _pad_to(lens, cp_size * ratio)
    cu = _cu(lens)
    compressor = _compressor(ratio)
    x = _packed_input(lens)
    reference, _ = compressor._forward_thd(x, cu)
    if reference is None:
        pytest.skip("no segment reaches the ratio, nothing to compare")
    got, _ = _cp_compress(compressor, x, cu, cp_size, max(lens))
    total_comp = int(compressed_cu_seqlens(cu, ratio)[-1])
    assert reference.shape[0] == total_comp
    torch.testing.assert_close(got[:total_comp], reference, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_chain_backward_matches_the_whole_stream(cp_size):
    """The compaction's scatter must send each source row's gradient back where it came from.

    Unlike the forward, this needs a tolerance: the scatter is an index_add_, and the two paths
    accumulate into a row in a different order.
    """
    ratio, lens = 4, [40, 24, 64]
    cu = _cu(lens)
    compressor = _compressor(ratio)
    base = _packed_input(lens).float()

    ref_x = base.clone().requires_grad_(True)
    compressor._forward_thd(ref_x, cu)[0].sum().backward()

    cp_x = base.clone().requires_grad_(True)
    _cp_compress(compressor, cp_x, cu, cp_size, max(lens))[0].sum().backward()

    torch.testing.assert_close(cp_x.grad, ref_x.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_segments_shorter_than_the_ratio_produce_no_group(cp_size):
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

    packed, cu_comp_tiny = _compressor(ratio)._forward_thd(_packed_input([5, 7]), _cu([5, 7]))
    assert packed is None
    assert cu_comp_tiny.tolist() == [0, 0, 0]


def test_pre_grouped_input_without_max_seqlen_is_rejected():
    """Pre-grouped input needs max_seqlen: its group ids address positions past the compacted
    row count, so falling back to x.size(0) would index out of the rope table.
    """
    ratio, lens = 4, [40, 24]
    cu = _cu(lens)
    c_cap = compact_group_capacity(sum(lens), ratio)
    _, comp_ids = compact_gather_index(cu, global_start=0, l_local=sum(lens), ratio=ratio, c_cap=c_cap)
    compact = torch.zeros(c_cap * ratio, 1, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_seqlen"):
        _compressor(ratio)._forward_thd(compact, cu, compressed_group_ids=comp_ids)
