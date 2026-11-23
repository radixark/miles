"""Qwen3.8-Flash-Next VJP reference: how TOP's captured QSA selection becomes the reference attention's mask."""

from types import SimpleNamespace

import pytest
import torch

from miles.utils.debug_utils.top_reference_qwen38 import _selected_keys, replayed_qsa


def test_packed_key_rows_become_a_document_local_mask():
    """Rows 5..8 of the packed stream are one document; -1 pads a row that selects fewer keys."""
    indices = torch.tensor([[5, -1, -1], [5, 6, -1], [5, 7, -1], [6, 7, 8]], dtype=torch.int32)
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(_selected_keys(indices, lo=5), expected)


@pytest.mark.parametrize(
    ("indices", "refusal"),
    [
        ([[5, 7, -1], [5, 6, -1], [5, 6, 7]], "not causal"),  # row 0 selects key 2, after itself
        ([[5, -1], [5, -1]], "misses its own row"),
        ([[4, 5], [5, 6]], "outside its document"),
        ([[5, -1], [5, 7]], "outside its document"),
    ],
)
def test_a_selection_the_reference_cannot_replay_is_refused(indices, refusal):
    """A mask built from such rows would silently replay another attention than TOP ran."""
    with pytest.raises(ValueError, match=refusal):
        _selected_keys(torch.tensor(indices, dtype=torch.int32), lo=5)


def test_the_reference_indexer_returns_tops_selection_and_is_restored():
    indexer = torch.nn.Identity()
    layer = SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))
    causal = torch.full((2, 2), torch.finfo(torch.float32).min).triu_(1)[None, None]
    with replayed_qsa(layer, torch.tensor([[3, -1], [3, 4]], dtype=torch.int32), lo=3):
        mask = indexer(None, None, causal, None)
    assert torch.equal(mask == 0, torch.tensor([[True, False], [True, True]])[None, None])
    assert "forward" not in vars(indexer)


def test_a_qsa_layer_without_a_captured_selection_is_refused():
    layer = SimpleNamespace(self_attn=SimpleNamespace(indexer=torch.nn.Identity()))
    with pytest.raises(ValueError, match="captured index selection"), replayed_qsa(layer, None, lo=0):
        pass


def test_a_linear_attention_layer_needs_no_selection():
    with replayed_qsa(SimpleNamespace(), None, lo=0):
        pass
