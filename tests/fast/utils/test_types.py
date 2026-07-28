"""Unit tests for Sample.strip_last_output_tokens."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
import pytest

from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def _make_sample(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    log_probs: bool = False,
    loss_mask: bool = False,
    routed_experts: bool = False,
    indexer_topk: bool = False,
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
    if indexer_topk:
        # shape: (num_tokens - 1, ...)
        s.rollout_indexer_topk = numpy.zeros((len(tokens) - 1, 2, 3), dtype=numpy.int32)
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

    def test_strip_truncates_indexer_topk(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], indexer_topk=True)
        original_len = len(s.rollout_indexer_topk)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_indexer_topk) == original_len - 2

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

    def test_strip_clips_weight_version_spans(self, tokenizer):
        """Stripping output tokens truncates overlapping spans and drops fully-stripped ones."""
        s = _make_sample([1, 2], [3, 4, 5, 6])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 6)]),
        ]
        s.strip_last_output_tokens(3, tokenizer)
        assert s.all_weight_version_spans == [WeightVersionSpan("v1", 2, 3)]
        assert len(s.weight_versions) == 2


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(sglang_speculative_algorithm=None)


def _make_meta_info(output_ids: list[int], **extra) -> dict:
    return {
        "finish_reason": {"type": "stop"},
        "completion_tokens": len(output_ids),
        "output_token_logprobs": [(-0.1, token_id) for token_id in output_ids],
        **extra,
    }


class TestWeightVersions:
    def test_update_from_meta_info_parses_per_token_weight_versions(self):
        """Per-token weight_versions from meta_info are shifted to absolute token indices."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(
            _make_args(),
            _make_meta_info(
                [3, 4, 5],
                weight_versions=[{"version": "v1", "start": 0, "end": 2}, {"version": "v2", "start": 2, "end": 3}],
            ),
        )
        assert s.weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4), WeightVersionSpan("v2", 4, 5)])
        ]

    def test_update_from_meta_info_synthesizes_span_from_scalar_weight_version(self):
        """Without per-token data, the scalar weight_version becomes one span over the new tokens."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4, 5], weight_version="v7"))
        assert s.weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("v7", 2, 5)])]

    def test_output_end_anchors_the_span_when_the_caller_appends_its_own_tokens(self):
        """output_end marks the end of the generated tokens, so filler stored after them is not covered."""
        meta = {"output_token_logprobs": [(-0.1, i) for i in range(4)], "weight_version": "v1"}
        call = WeightVersionsPerCall.from_meta_info(meta, output_end=14)
        assert call.spans == [WeightVersionSpan("v1", 10, 14)]

    def test_zero_length_spans_are_dropped(self):
        """An aborted call with no output tokens reports a zero-length span that covers nothing."""
        call = WeightVersionsPerCall.from_meta_info(
            {"output_token_logprobs": [], "weight_versions": [{"version": "v1", "start": 0, "end": 0}]}, output_end=7
        )
        assert call.spans == []

    def test_a_scalar_weight_version_without_logprobs_fails_instead_of_dropping_the_version(self):
        """Silently dropping the version would disable staleness checks for return_logprob=False callers."""
        with pytest.raises(AssertionError, match="requires return_logprob=True"):
            WeightVersionsPerCall.from_meta_info(
                {"completion_tokens": 3, "weight_version": "v1"},
                output_end=3,
            )

    def test_a_scalar_weight_version_on_an_empty_output_is_dropped_without_failing(self):
        """An aborted call that generated nothing has no output tokens to anchor the version to."""
        call = WeightVersionsPerCall.from_meta_info(
            {"completion_tokens": 0, "output_token_logprobs": [], "weight_version": "v1"},
            output_end=7,
        )
        assert call.spans == []

    def test_update_from_meta_info_records_a_call_without_weight_version(self):
        """A call the engine did not stamp still counts as one call, with no spans."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4, 5]))
        assert s.weight_versions == [WeightVersionsPerCall(spans=[])]
        assert s.all_weight_version_spans == []

    def test_turn_count_includes_calls_without_weight_versions(self):
        """The per-call nesting counts every generate call, stamped or not."""
        s = _make_sample([1, 2], [3, 4])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4]))
        s.tokens += [5, 6]
        s.response_length += 2
        s.update_from_meta_info(_make_args(), _make_meta_info([5, 6], weight_version="v2"))
        assert len(s.weight_versions) == 2
        assert s.all_weight_version_spans == [WeightVersionSpan("v2", 4, 6)]

    def test_update_from_meta_info_appends_one_entry_per_call(self):
        """Each generate call appends its own entry with correct absolute offsets."""
        s = _make_sample([1, 2], [3, 4])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4], weight_version="v1"))
        s.tokens += [5, 6, 7]
        s.response_length += 3
        s.update_from_meta_info(_make_args(), _make_meta_info([5, 6, 7], weight_version="v2"))
        assert s.weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 7)]),
        ]
        assert s.all_weight_version_spans == [WeightVersionSpan("v1", 2, 4), WeightVersionSpan("v2", 4, 7)]

    def test_reset_for_retry_clears_weight_versions(self):
        """reset_for_retry clears weight_versions along with other outputs."""
        s = _make_sample([1, 2], [3, 4])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)])]
        s.reset_for_retry()
        assert s.weight_versions == []

    def test_validate_accepts_contiguous_spans(self):
        """validate passes for ordered non-overlapping spans within the token range."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 5)]),
        ]
        s.validate()

    def test_validate_rejects_overlapping_spans_across_calls(self):
        """validate fails when spans from successive calls overlap."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 3, 5)]),
        ]
        with pytest.raises(AssertionError, match="invalid weight version span"):
            s.validate()

    def test_validate_rejects_span_beyond_tokens(self):
        """validate fails when a span extends past the token list."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 6)])]
        with pytest.raises(AssertionError, match="invalid weight version span"):
            s.validate()

    def test_validate_rejects_empty_version(self):
        """validate fails when a span carries an empty version string."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("", 2, 4)])]
        with pytest.raises(AssertionError, match="empty version"):
            s.validate()

    def test_to_dict_from_dict_roundtrip(self):
        """Per-call weight versions survive a to_dict/from_dict roundtrip as typed objects."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 5)]),
        ]
        restored = Sample.from_dict(s.to_dict())
        assert restored.weight_versions == s.weight_versions
        assert all(isinstance(span, WeightVersionSpan) for span in restored.all_weight_version_spans)

    def test_oldest_weight_version_reads_all_spans(self):
        """oldest_weight_version takes the minimum numeric version across every span."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("7", 2, 4), WeightVersionSpan("5", 4, 5)])]
        assert s.oldest_weight_version == 5
