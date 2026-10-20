import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torchtitan")

from miles.backends.torchtitan_utils.tpu_doc_attention import document_causal_attention, document_ids  # noqa: E402

# documents of 700, 900 and 1200 tokens, then a 1296-token padding tail: two cross a 1024 query block
_DOC_LENGTHS = [700, 900, 1200, 1296]


def _positions() -> torch.Tensor:
    return torch.cat([torch.arange(n) for n in _DOC_LENGTHS]).unsqueeze(0)


def test_document_ids_name_the_documents_the_position_resets_delimit():
    positions = _positions()
    starts = (positions == 0).cumsum(-1)
    ids = document_ids(positions)
    assert torch.equal(ids[:, 1:] != ids[:, :-1], starts[:, 1:] != starts[:, :-1])


@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_blocked_attention_matches_one_dense_document_causal_mask(num_kv_heads):
    """Each query block must see every earlier key of its own document, including across block boundaries."""
    torch.manual_seed(0)
    doc_ids = document_ids(_positions())
    seq_len = doc_ids.shape[1]
    q = torch.randn(1, 4, seq_len, 16, dtype=torch.float64)
    k = torch.randn(1, num_kv_heads, seq_len, 16, dtype=torch.float64)
    v = torch.randn(1, num_kv_heads, seq_len, 16, dtype=torch.float64)
    gqa = num_kv_heads != 4

    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    dense_mask = (doc_ids[:, :, None] == doc_ids[:, None, :]) & causal
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask.unsqueeze(1), enable_gqa=gqa)

    torch.testing.assert_close(document_causal_attention(q, k, v, doc_ids, enable_gqa=gqa), expected)
