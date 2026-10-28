from miles.rollout.generate_utils.generate_endpoint_utils import split_output_token_logprobs
from miles.utils.types import Sample


def _meta(items):
    return {"output_token_logprobs": items, "finish_reason": {"type": "stop"}}


def test_numeric_logprobs_pass_through():
    sample = Sample(index=0)
    tokens, log_probs = split_output_token_logprobs(sample, _meta([(-0.5, 11, None), (-0.25, 12, None)]))
    assert tokens == [11, 12]
    assert log_probs == [-0.5, -0.25]
    assert not sample.remove_sample
    assert "null_rollout_log_prob_positions" not in sample.metadata


def test_null_logprob_excludes_sample():
    sample = Sample(index=3, response_length=5)
    tokens, log_probs = split_output_token_logprobs(sample, _meta([(-0.5, 11, None), (None, 12, None)]))
    assert tokens == [11, 12]
    assert log_probs == [-0.5, 0.0]
    assert sample.remove_sample
    assert sample.metadata["null_rollout_log_prob_positions"] == [6]


def test_missing_field_returns_empty():
    sample = Sample(index=0)
    assert split_output_token_logprobs(sample, {"finish_reason": {"type": "abort"}}) == ([], [])
