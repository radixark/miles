import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torchtitan")

from miles.backends.torchtitan_utils.sdpa_document_attention import (  # noqa: E402
    BlockedSdpaInnerAttention,
    document_causal_attention,
    document_ids,
)

# documents of 700, 900 and 1200 tokens, then a 1296-token padding tail: two cross a 1024 query block
_DOC_LENGTHS = [700, 900, 1200, 1296]


def _positions() -> torch.Tensor:
    return torch.cat([torch.arange(n) for n in _DOC_LENGTHS])


def _dense_reference(q, k, v, doc_ids, *, enable_gqa):
    seq_len = doc_ids.shape[-1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    mask = (doc_ids[:, :, None] == doc_ids[:, None, :]) & causal
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), enable_gqa=enable_gqa)


def test_document_ids_name_the_documents_the_position_resets_delimit():
    positions = _positions()
    starts = (positions == 0).cumsum(-1)
    ids = document_ids(positions)
    assert ids.dtype == torch.int32
    assert torch.equal(ids[1:] != ids[:-1], starts[1:] != starts[:-1])


@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_blocked_attention_matches_one_dense_document_causal_mask(num_kv_heads):
    """Each query block must see every earlier key of its own document, including across block boundaries."""
    torch.manual_seed(0)
    doc_ids = document_ids(_positions()).unsqueeze(0)
    seq_len = doc_ids.shape[1]
    q = torch.randn(1, 4, seq_len, 16, dtype=torch.float64)
    k = torch.randn(1, num_kv_heads, seq_len, 16, dtype=torch.float64)
    v = torch.randn(1, num_kv_heads, seq_len, 16, dtype=torch.float64)
    gqa = num_kv_heads != 4

    expected = _dense_reference(q, k, v, doc_ids, enable_gqa=gqa)
    torch.testing.assert_close(document_causal_attention(q, k, v, doc_ids, enable_gqa=gqa), expected)


def test_the_inner_attention_takes_torchtitans_token_major_layout():
    """torchtitan hands inner attention [T, H, K] tensors and the metadata its config built from the positions."""
    torch.manual_seed(0)
    config = BlockedSdpaInnerAttention.Config()
    attention = config.build()
    positions = _positions()
    seq_len = positions.shape[0]
    q, k, v = (torch.randn(seq_len, heads, 16, dtype=torch.float64) for heads in (4, 2, 2))

    out = attention(q, k, v, attention_metadata=config.build_attention_metadata(positions), enable_gqa=True)

    to_bhtk = lambda t: t.transpose(0, 1).unsqueeze(0)  # noqa: E731
    expected = _dense_reference(
        to_bhtk(q), to_bhtk(k), to_bhtk(v), document_ids(positions).unsqueeze(0), enable_gqa=True
    )
    torch.testing.assert_close(out, expected.squeeze(0).transpose(0, 1))


def test_the_inner_attention_refuses_what_it_cannot_honor():
    attention = BlockedSdpaInnerAttention.Config().build()
    q = torch.randn(8, 2, 4)
    doc_ids = document_ids(torch.arange(8))
    with pytest.raises(NotImplementedError):
        attention(q, q, q, attention_metadata=doc_ids, score_mod=lambda score, *idx: score)
    with pytest.raises(NotImplementedError):
        attention(q, q, q, attention_metadata=doc_ids, out_transform=lambda out, lse: out)
    with pytest.raises(TypeError, match="document ids"):
        attention(q, q, q, attention_metadata=object())
