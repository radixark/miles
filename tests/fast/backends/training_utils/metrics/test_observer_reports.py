"""A rank's per-token training values say which response positions they cover, so readers need not know the CP split."""

import numpy as np
import pytest
import torch

from miles.backends.training_utils.data.context_parallel import (
    assemble_log_prob_from_cp,
    get_logits_and_tokens_offset_with_cp,
)
from miles.backends.training_utils.metrics.observer_reports import response_positions, train_token_rows

LAYOUTS = [
    # total_length, response_length, cp_size, qkv_format, max_seq_len
    (37, 20, 2, "thd", None),
    (64, 63, 4, "thd", None),
    (9, 2, 4, "thd", None),
    (40, 25, 2, "bshd", 48),
]


def _rank_slice(values: torch.Tensor, *, total_length: int, cp_rank: int, cp_size: int, qkv_format, max_seq_len):
    # the same cut the forward pass makes
    prompt_length = total_length - len(values)
    _, _, logits_offsets, _ = get_logits_and_tokens_offset_with_cp(
        total_length, len(values), qkv_format, max_seq_len, cp_rank=cp_rank, cp_size=cp_size
    )
    return torch.cat([values[lo - (prompt_length - 1) : hi - (prompt_length - 1)] for lo, hi in logits_offsets])


@pytest.mark.parametrize("total_length,response_length,cp_size,qkv_format,max_seq_len", LAYOUTS, ids=str)
def test_positions_put_every_rank_slice_where_reassembly_does(
    total_length, response_length, cp_size, qkv_format, max_seq_len
):
    values = torch.randn(response_length)
    chunks = {
        cp_rank: _rank_slice(
            values,
            total_length=total_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
        )
        for cp_rank in range(cp_size)
    }

    placed = np.full(response_length, np.nan)
    for cp_rank, chunk in chunks.items():
        positions = response_positions(
            total_length,
            response_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
        )
        assert len(positions) == len(chunk)
        assert np.isnan(placed[positions]).all(), "two ranks claim the same position"
        placed[positions] = chunk.numpy()

    expected = assemble_log_prob_from_cp(
        chunks, total_length, response_length, cp_size, qkv_format=qkv_format, max_seq_len=max_seq_len
    )
    np.testing.assert_array_equal(placed, expected.numpy())


def test_rows_without_context_parallelism_cover_the_whole_response():
    rollout_data = dict(
        sample_indices=[4, 9],
        total_lengths=[6, 5],
        response_lengths=[3, 2],
        rewards=[0.5, -0.5],
        truncated=[0, 1],
        loss_masks=[torch.tensor([1, 1, 0]), torch.tensor([1, 1])],
        log_probs=[torch.tensor([-1.0, -2.0, -3.0]), torch.tensor([-0.5, -0.25])],
        # a slice that does not fit the layout is dropped, not misplaced
        entropy=[torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
    )

    rows = train_token_rows(rollout_data, rank=2, cp_rank=0, cp_size=1, qkv_format="thd")

    assert [(r["sample_index"], r["rank"], r["truncated"]) for r in rows] == [(4, 2, False), (9, 2, True)]
    np.testing.assert_array_equal(rows[0]["positions"], [0, 1, 2])
    np.testing.assert_array_equal(rows[0]["log_probs"], [-1.0, -2.0, -3.0])
    assert rows[0]["entropy"] is None
    np.testing.assert_array_equal(rows[1]["entropy"], np.array([0.3, 0.4], dtype=np.float32))
    assert rows[0]["advantages"] is None
