"""Trie attention on CPU, with flash-attn replaced by the plain-PyTorch trie reference: each segment must attend over
the keys its layout gathers, and calls it cannot serve over a trie row must be refused.
``tests/fast-gpu/test_fsdp_trie_packing.py`` runs the real kernels.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.reference_trie_attention import reference_trie_attention

from miles.backends.fsdp_utils.adaptations.trie_attention import (
    TrieAttentionArgs,
    trie_attention_forward,
    trie_model_spec,
)
from miles.backends.training_utils.data.trie_packing import TrieKV, build_trie_layout, pack_trie_bins


def _reference_flash_attn_varlen(
    q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, window_size, softmax_scale
):
    """flash_attn_varlen_func's contract, bottom-right causal, on the trie reference."""
    assert causal
    kv = TrieKV(index=torch.arange(k.size(0)), cu_seqlens=cu_seqlens_k, max_seqlen=max_seqlen_k)
    window_left = None if window_size[0] < 0 else window_size[0]
    return reference_trie_attention(q, k, v, kv, cu_seqlens_q, window_left=window_left, softmax_scale=softmax_scale)


@pytest.fixture
def reference_flash_attn(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "flash_attn", SimpleNamespace(flash_attn_varlen_func=_reference_flash_attn_varlen)
    )


def _trie_row(window_left: int | None = None):
    """Three samples sharing a 4-token prefix, two of them 2 more, and a sample of its own, in one padded row."""
    tokens = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 8, 9], [1, 2, 3, 4, 8, 10, 11], [12, 13]]
    (trie,) = pack_trie_bins(tokens, max_bin_width=64)
    layout = build_trie_layout(trie, window_left=window_left, window_start_multiple=2, pad_multiple=8)
    return layout, TrieAttentionArgs(layout, torch.device("cpu"))


@pytest.mark.parametrize("window", [None, 3], ids=["full", "sliding"])
def test_each_segment_attends_over_its_gathered_keys(reference_flash_attn, window):
    window_left = None if window is None else window - 1
    layout, trie = _trie_row(window_left)
    if window is not None:
        assert layout.window_kv.index.numel() < layout.kv.index.numel(), "no window gather dropped an ancestor"
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, layout.num_tokens, 8) for heads in (4, 2, 2))

    got, _ = trie_attention_forward(None, q, k, v, None, sliding_window=window, miles_trie_attn_args=trie)

    kv = layout.kv if window is None else layout.window_kv
    want = reference_trie_attention(
        *(x[0].transpose(0, 1) for x in (q, k, v)), kv, layout.cu_seqlens_q, window_left=window_left
    )
    torch.testing.assert_close(got[0], want)


def test_a_layer_type_that_would_mix_across_samples_is_refused():
    config = SimpleNamespace(layer_types=["full_attention", "chunked_attention"], sliding_window=None)
    with pytest.raises(ValueError, match="chunked_attention"):
        trie_model_spec(config)


def test_a_call_without_the_trie_row_is_refused(reference_flash_attn):
    q = k = v = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="miles_trie_attn_args"):
        trie_attention_forward(None, q, k, v, None)


def test_a_sliding_window_outside_layer_types_is_refused():
    """Some models' layers pass ``sliding_window`` with no sliding layer type, so rows would lack its gather."""
    config = SimpleNamespace(layer_types=["full_attention"], sliding_window=8)
    with pytest.raises(ValueError, match="sliding_window=8"):
        trie_model_spec(config)


@pytest.mark.parametrize("argument", ["s_aux", "softcap"])
def test_attention_arguments_flash_attn_cannot_honor_are_refused(reference_flash_attn, argument):
    layout, trie = _trie_row()
    q = k = v = torch.zeros(1, 2, layout.num_tokens, 8)
    with pytest.raises(ValueError, match=argument):
        trie_attention_forward(None, q, k, v, None, miles_trie_attn_args=trie, **{argument: torch.ones(2)})
