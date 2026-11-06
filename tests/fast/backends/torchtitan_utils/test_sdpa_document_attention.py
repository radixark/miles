import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from tests.fast.backends.torchtitan_utils.test_document_attention import _dense_reference, _positions

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])
pytest.importorskip("torchtitan")

from miles.backends.torchtitan_utils.document_attention import document_ids  # noqa: E402
from miles.backends.torchtitan_utils.sdpa_document_attention import BlockedSdpaInnerAttention  # noqa: E402


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
