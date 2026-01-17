from unittest.mock import MagicMock

import pytest

from miles.rollout.generate_hub.sample_utils import merge_sample_pair
from miles.utils.types import Sample


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode = lambda tokens: f"<decoded:{tokens}>"
    return tokenizer


def make_sample(
    prompt="test_prompt",
    tokens=None,
    response="",
    response_length=0,
    loss_mask=None,
    rollout_log_probs=None,
    status=Sample.Status.COMPLETED,
    label="test_label",
    reward=1.0,
    index=0,
    group_index=0,
):
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        status=status,
        label=label,
        reward=reward,
        index=index,
        group_index=group_index,
    )


class TestMergeSamples:
    def test_basic_merge(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 3, 10, 11, 12],
            response="response1",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.1, -0.2, -0.3],
        )
        b = make_sample(
            tokens=[1, 2, 3, 10, 11, 12, 20, 21, 30, 31, 32],
            response="response2",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.4, -0.5, -0.6],
            status=Sample.Status.TRUNCATED,
        )

        merged = merge_sample_pair(a, b, mock_tokenizer)

        assert merged.tokens == b.tokens
        assert merged.response_length == 3 + 2 + 3
        assert merged.loss_mask == [1, 1, 1, 0, 0, 1, 1, 1]
        assert merged.rollout_log_probs == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5, -0.6]
        assert merged.prompt == a.prompt
        assert merged.status == b.status
        assert merged.label == a.label
        assert merged.index == a.index
        assert merged.group_index == a.group_index
        assert "response1" in merged.response
        assert "response2" in merged.response
        assert "<decoded:[20, 21]>" in merged.response

    def test_loss_mask_none_defaults_to_all_ones(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=None,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=None,
        )

        merged = merge_sample_pair(a, b, mock_tokenizer)

        assert merged.loss_mask == [1, 0, 1]
        assert merged.rollout_log_probs == [0.0, 0.0, 0.0]

    def test_tokens_prefix_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 3],
            response_length=1,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 99, 20, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_sample_pair(a, b, mock_tokenizer)

    def test_field_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            index=0,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            index=1,
        )

        with pytest.raises(AssertionError, match="index mismatch"):
            merge_sample_pair(a, b, mock_tokenizer)

    def test_obs_len_invalid_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="obs_len must be > 0"):
            merge_sample_pair(a, b, mock_tokenizer)

    def test_sample_validate_fails_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10, 11],
            response_length=2,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 11, 20, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="loss_mask length"):
            merge_sample_pair(a, b, mock_tokenizer)
