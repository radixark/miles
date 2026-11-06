"""The TPU flash attention must equal one dense causal mask within each segment."""

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

from miles.kernels.tpu.flash_attention import flash_attention

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(est_time=10, suite="stage-b-2-gpu-h200", labels=["torchtitan"], hardware=["hopper"], num_gpus=2)

# segments of 700, 900 and 1200 tokens, then a 1296-token padding tail: two cross a 1024-query block
_SEGMENT_LENGTHS = [700, 900, 1200, 1296]


def _positions() -> torch.Tensor:
    return torch.cat([torch.arange(n) for n in _SEGMENT_LENGTHS])


def _segment_ids() -> torch.Tensor:
    positions = _positions()
    return (torch.arange(positions.shape[0]) - positions).unsqueeze(0)


def _dense_reference(q, k, v, segment_ids, *, enable_gqa):
    seq_len = segment_ids.shape[-1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    mask = (segment_ids[:, :, None] == segment_ids[:, None, :]) & causal
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), enable_gqa=enable_gqa)


@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_blocks_match_one_dense_mask(num_kv_heads):
    """Each query block must see every earlier key of its own segment, including across block boundaries."""
    torch.manual_seed(0)
    segment_ids = _segment_ids()
    seq_len = segment_ids.shape[1]
    q = torch.randn(1, 4, seq_len, 16, dtype=torch.float64)
    k = torch.randn(1, num_kv_heads, seq_len, 16, dtype=torch.float64)
    v = torch.randn(1, num_kv_heads, seq_len, 16, dtype=torch.float64)
    gqa = num_kv_heads != 4

    expected = _dense_reference(q, k, v, segment_ids, enable_gqa=gqa)
    torch.testing.assert_close(flash_attention(q, k, v, segment_ids=segment_ids, enable_gqa=gqa), expected)


def test_the_torchtitan_inner_attention_takes_token_major_tensors():
    """torchtitan hands inner attention [T, H, D] tensors and the segment ids its config built from the positions."""
    from miles.backends.torchtitan_utils.config import TpuFlashInnerAttention

    torch.manual_seed(0)
    config = TpuFlashInnerAttention.Config()
    attention = config.build()
    positions = _positions()
    segment_ids = config.build_attention_metadata(positions)
    assert segment_ids.dtype == torch.int32
    # a new segment starts exactly where the positions restart
    starts = (positions == 0).cumsum(-1)
    assert torch.equal(segment_ids[1:] != segment_ids[:-1], starts[1:] != starts[:-1])
    q, k, v = (torch.randn(positions.shape[0], heads, 16, dtype=torch.float64) for heads in (4, 2, 2))

    out = attention(q, k, v, attention_metadata=segment_ids, enable_gqa=True)

    to_bhtd = lambda t: t.transpose(0, 1).unsqueeze(0)  # noqa: E731
    expected = _dense_reference(to_bhtd(q), to_bhtd(k), to_bhtd(v), _segment_ids(), enable_gqa=True)
    torch.testing.assert_close(out, expected.squeeze(0).transpose(0, 1))
