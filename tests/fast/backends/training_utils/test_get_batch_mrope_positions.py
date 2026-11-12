"""`get_batch` places multimodal rows and mRoPE positions where each CP rank's packed tokens are."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.common.embeddings import rope_utils

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils


def _get_batch(monkeypatch, rollout, *, cp_rank, cp_size):
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    return data_utils.get_batch(
        data_utils.DataIterator(rollout, micro_batch_size=len(rollout["tokens"])),
        list(rollout),
        pad_multiplier=7,
        qkv_format="thd",
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_every_cp_rank_rotates_each_token_with_its_own_position(monkeypatch, cp_size):
    """Each token's id encodes its sample and index, and so does the position SGLang gave it."""
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    # Return the looked-up frequency rows instead of rotating with them.
    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_bshd", lambda t, freqs, **kwargs: freqs)
    lengths = [13, 5, 9]
    token_ids = [1000 * (i + 1) + torch.arange(n) for i, n in enumerate(lengths)]
    rollout = {
        "tokens": token_ids,
        "mrope_position_ids": [ids.unsqueeze(1).expand(-1, 3) for ids in token_ids],
        "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n // 2 for n in lengths],
    }

    for cp_rank in range(cp_size):
        batch = _get_batch(monkeypatch, rollout, cp_rank=cp_rank, cp_size=cp_size)

        position_ids = batch["position_ids"]
        assert "mrope_position_ids" not in batch
        assert position_ids.shape == (3, 1, batch["cu_seqlens_host"][-1])
        # Megatron's mRoPE frequency table has one row per natural token, here just its T position.
        freqs = position_ids[0, 0].to(torch.float32).view(-1, 1, 1, 1)
        tokens = batch["tokens"][0]
        rows = rope_utils._apply_rotary_pos_emb_thd(
            torch.zeros(tokens.numel(), 1, 1),
            batch["cu_seqlens"],
            freqs,
            cp_group=SimpleNamespace(size=lambda: cp_size, rank=lambda rank=cp_rank: rank),
            max_seqlen=batch["max_seqlen"],
        ).view(-1)
        real = tokens != 0
        assert torch.equal(rows[real].long(), tokens[real])


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_tower_rows_land_on_the_rank_holding_their_token(monkeypatch, cp_size):
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    lengths = [13, 5, 9]
    token_ids = [1000 * (i + 1) + torch.arange(n) for i, n in enumerate(lengths)]
    image_positions = [torch.tensor([2, 3, 4, 11]), None, torch.tensor([0, 8])]
    rollout = {
        "tokens": token_ids,
        "multimodal_train_inputs": [
            None if positions is None else {"mm_image_positions": positions} for positions in image_positions
        ],
        "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n // 2 for n in lengths],
    }
    # The token every tower row belongs to, in the order the rows are concatenated.
    expected = torch.cat([token_ids[i][p] for i, p in enumerate(image_positions) if p is not None])

    placed = torch.zeros_like(expected)
    times_placed = torch.zeros_like(expected)
    for cp_rank in range(cp_size):
        batch = _get_batch(monkeypatch, rollout, cp_rank=cp_rank, cp_size=cp_size)

        rows = batch["multimodal_train_inputs"]["mm_image_positions"]
        on_rank = rows >= 0
        placed[on_rank] = batch["tokens"][0, rows[on_rank]]
        times_placed += on_rank

    # Every row is placed exactly once across the CP ranks, at its own token.
    assert times_placed.eq(1).all()
    assert torch.equal(placed, expected)
