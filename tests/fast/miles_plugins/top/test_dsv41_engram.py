"""DeepSeek-V4.1 TOP Engram: what the trainer binding refuses instead of hashing another program's n-grams."""

import pytest
import torch

from miles_plugins.top.dsv41_engram import packed_hash_ids, top_engram_gate


def test_hashing_refuses_lengths_that_do_not_cover_the_tokens():
    """A request boundary in the wrong place shifts every later n-gram."""
    with pytest.raises(NotImplementedError, match="cover every token"):
        packed_hash_ids(None, torch.zeros(5, dtype=torch.int64), [2, 2], 0)


def test_hashing_refuses_batched_ids():
    with pytest.raises(NotImplementedError, match="packed requests"):
        packed_hash_ids(None, torch.zeros(2, 3, dtype=torch.int64), [3, 3], 0)


def test_the_gate_refuses_flat_streams():
    with pytest.raises(NotImplementedError, match="streams"):
        top_engram_gate(torch.zeros(2, 32), torch.zeros(2, 40), torch.ones(4, 8), torch.ones(4, 8), 1e-20, 1e-6)
