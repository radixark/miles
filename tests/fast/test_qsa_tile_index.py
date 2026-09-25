"""The QSA tile map must list every key tile a query can attend, including packs off the block grid."""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_block_sparse_attn import (  # noqa: E402
    build_tile_index,
    build_tile_index_pair,
)
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import PackedBlockLayout  # noqa: E402

RATIO = 4
BQ = 64

_rng = random.Random(0)
PACKS = [
    [61, 128],
    [61, 61, 61, 200],
    [64, 128],
    [1, 1, 1, 1, 1, 200],
    *[[_rng.randint(1, 300) for _ in range(_rng.randint(2, 6))] for _ in range(4)],
]


def _packed_case(lens, dense: bool, seed: int = 0):
    """Layout plus a causal, in-sequence block selection; ``dense`` selects every allowed block."""
    cu = torch.tensor([0, *lens]).cumsum(0)
    total = int(cu[-1])
    starts = cu[:-1]
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    positions = torch.arange(total) - starts[seg]
    layout = PackedBlockLayout(cu, positions, RATIO)

    g = torch.Generator().manual_seed(seed)
    sel = torch.zeros(total, layout.num_blocks, dtype=torch.uint8)
    for t in range(total):
        own = int(layout.token_block_start[t])
        allowed = torch.arange(own, own + int(positions[t]) // RATIO + 1)
        if not dense:
            allowed = allowed[torch.rand(allowed.numel(), generator=g) < 0.3]
            allowed = torch.cat([allowed, torch.tensor([own + int(positions[t]) // RATIO])])
        sel[t, allowed] = 1
    return layout, positions, sel


def _attended_keys(t, layout, positions, sel):
    """Keys the kernel mask lets query ``t`` see: block selected, causal, same sequence."""
    tok_base = int(layout.token_start[t])
    blk_base = int(layout.token_block_start[t])
    keys = torch.arange(tok_base, t + 1)
    blocks = blk_base + (keys - tok_base) // RATIO
    return keys[sel[t, blocks] != 0]


def _tile_sets(lst, cnt):
    return [set(lst[i, : int(cnt[i])].tolist()) for i in range(cnt.numel())]


@pytest.mark.parametrize("lens", PACKS, ids=lambda lens: "-".join(map(str, lens)))
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "sparse"])
def test_every_attended_key_tile_is_listed(lens, dense):
    layout, positions, sel = _packed_case(lens, dense)
    first, last = layout.block_first_token, layout.block_last_token
    total = sel.shape[0]

    fwd = _tile_sets(*build_tile_index(sel, first, last, BQ, 64))
    klist, kcnt, qlist, qcnt = build_tile_index_pair(sel, first, last, BQ, 32)
    bwd_k = _tile_sets(klist, kcnt)
    bwd_q = _tile_sets(qlist, qcnt)

    assert len(bwd_q) == -(-total // 32)
    for t in range(total):
        keys = _attended_keys(t, layout, positions, sel)
        missing_fwd = sorted({int(k) for k in keys if int(k) // 64 not in fwd[t // BQ]})
        missing_bwd = sorted({int(k) for k in keys if int(k) // 32 not in bwd_k[t // BQ]})
        assert not missing_fwd, f"query {t}: forward tile list misses keys {missing_fwd}"
        assert not missing_bwd, f"query {t}: backward tile list misses keys {missing_bwd}"

    forward_pairs = {(qt, kt) for qt, tiles in enumerate(bwd_k) for kt in tiles}
    backward_pairs = {(qt, kt) for kt, tiles in enumerate(bwd_q) for qt in tiles}
    assert forward_pairs == backward_pairs


def test_block_spans_stay_inside_their_sequence():
    layout, _, _ = _packed_case([61, 128, 3], dense=True)
    cu = torch.tensor([0, 61, 189, 192])
    seq = layout.block_seq
    assert torch.equal(layout.block_first_token, cu[seq] + RATIO * layout.block_local)
    assert bool((layout.block_last_token < cu[seq + 1]).all())
    assert bool((layout.block_last_token - layout.block_first_token < RATIO).all())
