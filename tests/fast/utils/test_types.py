"""Unit tests for Sample.strip_last_output_tokens."""

from unittest.mock import MagicMock

import numpy
import pytest

from miles.utils.types import Sample


def _make_sample(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    log_probs: bool = False,
    loss_mask: bool = False,
    routed_experts: bool = False,
) -> Sample:
    """Create a Sample with the given prompt + response token IDs."""
    tokens = prompt_ids + response_ids
    s = Sample(
        tokens=tokens,
        response_length=len(response_ids),
        response="dummy",
    )
    if log_probs:
        s.rollout_log_probs = [-0.1] * len(response_ids)
    if loss_mask:
        s.loss_mask = [1] * len(response_ids)
    if routed_experts:
        # shape: (num_tokens - 1, ...)
        s.rollout_routed_experts = numpy.zeros((len(tokens) - 1, 2, 2), dtype=numpy.int32)
    return s


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.decode = lambda ids: "".join(chr(65 + i) for i in ids)
    return tok


class TestStripLastOutputTokens:
    def test_strip_zero_is_noop(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(0, tokenizer)
        assert s.tokens == original_tokens
        assert s.response_length == 3

    def test_strip_basic(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(2, tokenizer)
        assert s.tokens == [1, 2, 3]
        assert s.response_length == 1

    def test_strip_all_response(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(3, tokenizer)
        assert s.tokens == [1, 2]
        assert s.response_length == 0
        assert s.response == ""

    def test_strip_too_many_raises(self, tokenizer):
        s = _make_sample([1, 2], [3, 4])
        with pytest.raises(AssertionError, match="cannot strip 3 tokens"):
            s.strip_last_output_tokens(3, tokenizer)

    def test_strip_truncates_log_probs(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], log_probs=True)
        assert len(s.rollout_log_probs) == 3
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_log_probs) == 1

    def test_strip_truncates_loss_mask(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], loss_mask=True)
        assert len(s.loss_mask) == 3
        s.strip_last_output_tokens(1, tokenizer)
        assert len(s.loss_mask) == 2

    def test_strip_truncates_routed_experts(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], routed_experts=True)
        original_len = len(s.rollout_routed_experts)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_routed_experts) == original_len - 2

    def test_strip_updates_response_text(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(1, tokenizer)
        # response should be re-decoded from the remaining response tokens
        assert s.response == tokenizer.decode(s.tokens[-s.response_length :])

    def test_strip_negative_is_noop(self, tokenizer):
        s = _make_sample([1, 2], [3, 4])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(-1, tokenizer)
        assert s.tokens == original_tokens
