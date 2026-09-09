"""QSA indexer parity between a packed (thd) batch and one-sequence-at-a-time.

sglang scores one request at a time, so its compressed-block grid always starts at
that request's token 0. Training packs several sequences into one micro-batch, and
the indexer's positions restart per sequence while the block grid used to be global
over the whole buffer -- so every sequence after the first scored the blocks at the
front of the buffer, had them clamped away by the caller, and ended up attending to
nothing but its own tail. It cost 1.5-4.8 nats of train/rollout logprob gap on every
such sample (position 0 in the micro-batch was always fine, which is what made it
look like a "short sample" problem).

These tests pin the packed path to the per-sequence result, with a boundary that is
deliberately NOT a multiple of the compress ratio.
"""

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import (  # noqa: E402
    PackedBlockLayout,
    block_causal_mask,
    compress_keys_by_mean,
    compress_keys_by_mean_packed,
    packed_block_causal_mask,
)

RATIO = 4
LENS = [37, 13, 22]  # none is a multiple of RATIO
DIM = 8


def _packed_positions(cu, total):
    idx = torch.arange(total)
    starts = cu[:-1].long()
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    return idx - starts[seg]


def _fixture():
    cu = torch.tensor([0, *LENS]).cumsum(0).to(torch.int32)
    total = int(cu[-1])
    positions = _packed_positions(cu, total)
    torch.manual_seed(0)
    token_k = torch.randn(total, DIM)
    return cu, total, positions, token_k


def test_packed_blocks_never_mix_sequences():
    cu, total, positions, token_k = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    packed = compress_keys_by_mean_packed(token_k, layout)

    expected = torch.cat(
        [compress_keys_by_mean(token_k[int(cu[i]) : int(cu[i + 1])], RATIO) for i in range(len(LENS))],
        dim=0,
    )
    assert packed.shape == expected.shape
    torch.testing.assert_close(packed, expected)


def test_packed_mask_matches_per_sequence_mask():
    cu, total, positions, _ = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    packed = packed_block_causal_mask(positions, layout, RATIO)
    assert packed.shape == (total, layout.num_blocks)

    block_start = 0
    for i, length in enumerate(LENS):
        lo, hi = int(cu[i]), int(cu[i + 1])
        n_blocks = -(-length // RATIO)
        alone = block_causal_mask(positions[lo:hi], n_blocks, RATIO)

        window = packed[lo:hi, block_start : block_start + n_blocks]
        torch.testing.assert_close(window, alone)

        # nothing outside the sequence's own block range may be visible
        outside = packed[lo:hi].clone()
        outside[:, block_start : block_start + n_blocks] = False
        assert not bool(outside.any()), f"sequence {i} can see blocks of another sequence"
        block_start += n_blocks


def test_global_block_grid_points_at_the_wrong_sequence():
    """Characterises the bug the packed path exists to avoid.

    A global block grid combined with per-sequence positions sends every sequence
    after the first to the blocks at the front of the buffer, and hides its own.
    """
    cu, total, positions, _ = _fixture()
    num_blocks_global = -(-total // RATIO)
    global_mask = block_causal_mask(positions, num_blocks_global, RATIO)

    lo, hi = int(cu[1]), int(cu[2])  # the second sequence
    own_lo = lo // RATIO
    row = hi - 1  # its last query, which has the most history
    visible = global_mask[row].nonzero().flatten()

    assert visible.numel() > 0
    assert int(visible.max()) < own_lo, "expected the global grid to look before the sequence start"
    assert not bool(global_mask[row, own_lo:].any()), "expected the sequence's own blocks to be hidden"


def test_layout_block_local_positions_restart_per_sequence():
    cu, total, positions, _ = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    expected = torch.cat([torch.arange(-(-length // RATIO)) for length in LENS])
    torch.testing.assert_close(layout.block_local, expected)
    expected_starts = torch.cat(
        [int(cu[i]) + torch.arange(-(-length // RATIO)) * RATIO for i, length in enumerate(LENS)]
    )
    torch.testing.assert_close(layout.block_token_start, expected_starts)
    assert layout.num_blocks == sum(-(-length // RATIO) for length in LENS)


@pytest.mark.parametrize("lens", [[37, 13], [64, 64], [1, 41]])
def test_selection_equals_single_sequence_runs(lens):
    """Full indexer: packed selection == per-sequence selection + token offset."""
    if not torch.cuda.is_available():
        pytest.skip("indexer projection needs a CUDA device (TELinear)")

    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer

    hidden = 64
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    config.qwen3_8_next_indexer_n_heads = 2
    config.qwen3_8_next_indexer_kv_heads = 1
    config.qwen3_8_next_indexer_head_dim = 32
    config.qwen3_8_next_indexer_budget = 16
    config.qwen3_8_next_indexer_compress_ratio = RATIO

    indexer = Qwen38NextQSAIndexer(config, layer_number=1).cuda()
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32).cuda()
    total = int(cu[-1])
    positions = _packed_positions(cu.cpu(), total).cuda()
    torch.manual_seed(0)
    hidden_states = torch.randn(total, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        packed = indexer(hidden_states, positions, cu_seqlens=cu)
        for i, length in enumerate(lens):
            lo = int(cu[i])
            sub_cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
            alone = indexer(
                hidden_states[lo : lo + length].contiguous(),
                positions[lo : lo + length].contiguous(),
                cu_seqlens=sub_cu,
            )
            shifted = torch.where(alone >= 0, alone + lo, alone)
            got = packed[lo : lo + length]

            # The returned width is min(block_topk, num_blocks) * ratio, so a sequence
            # scored alone can come back narrower than the same sequence inside a pack.
            # What has to match is the set of key tokens each query may look at.
            for row in range(length):
                got_set = {int(t) for t in got[row].tolist() if t >= 0}
                want_set = {int(t) for t in shifted[row].tolist() if t >= 0}
                assert got_set == want_set, (
                    f"sequence {i} row {row}: packed selection {sorted(got_set)} "
                    f"!= single-sequence {sorted(want_set)}"
                )
