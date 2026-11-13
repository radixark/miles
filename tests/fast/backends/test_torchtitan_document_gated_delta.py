import pytest
import torch

pytest.importorskip("torchtitan")

from miles.backends.torchtitan_utils.document_gated_delta import DocumentGatedDeltaInnerAttention  # noqa: E402


def test_the_inner_attention_runs_convs_gate_and_delta_rule_per_document():
    """Packing documents must not change any document's output: each one run alone gives the same rows."""
    torch.manual_seed(0)
    config = DocumentGatedDeltaInnerAttention.Config()
    attention = config.build()
    doc_lengths = [5, 70, 3]
    seq_len = sum(doc_lengths)

    def run(start: int, length: int, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        sliced = {name: t[start : start + length] if name in _TOKEN_INPUTS else t for name, t in tensors.items()}
        return attention(
            **sliced,
            key_head_dim=8,
            value_head_dim=8,
            attention_metadata=config.build_attention_metadata(torch.arange(length)),
        )

    tensors = dict(
        query_TC=torch.randn(seq_len, 16),
        key_TC=torch.randn(seq_len, 16),
        value_TC=torch.randn(seq_len, 32),
        a_TH=torch.randn(seq_len, 4),
        b_TH=torch.randn(seq_len, 4),
        conv_q_weight_C1W=torch.randn(16, 1, 4),
        conv_k_weight_C1W=torch.randn(16, 1, 4),
        conv_v_weight_C1W=torch.randn(32, 1, 4),
        A_log_H=torch.randn(4),
        dt_bias_H=torch.randn(4),
    )
    packed = attention(
        **tensors,
        key_head_dim=8,
        value_head_dim=8,
        attention_metadata=config.build_attention_metadata(_positions(doc_lengths)),
    )
    starts = torch.tensor([0, *doc_lengths]).cumsum(0).tolist()
    alone = torch.cat([run(start, length, tensors) for start, length in zip(starts[:-1], doc_lengths, strict=True)])
    torch.testing.assert_close(packed, alone, atol=1e-5, rtol=1e-4)


def _positions(doc_lengths: list[int]) -> torch.Tensor:
    return torch.cat([torch.arange(n) for n in doc_lengths])


_TOKEN_INPUTS = {"query_TC", "key_TC", "value_TC", "a_TH", "b_TH"}
