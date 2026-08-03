"""CPU unit tests for the DeepSeek-V4 THD (packed varlen) index layer.

Packing samples into one stream, and splitting that stream across CP ranks, must give each
sample exactly the rows it would get on its own. The sparse-attention kernel takes no
`cu_seqlens`, so `topk_idxs` alone decides what a query can see.

The compressor lives in tests/manual/models/deepseek_v4/, which a CPU image cannot run, and
the CP collectives in tests/e2e/precision/test_dsv4_thd_cp_correctness.py.
"""

import random

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

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

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

WINDOW = 128

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
