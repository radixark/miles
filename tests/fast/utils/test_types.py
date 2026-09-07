"""Unit tests for Sample.strip_last_output_tokens."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
import pytest

from miles.utils.types import LEGACY_WEIGHT_VERSIONS_KEY, Sample, WeightVersionSpan, WeightVersionsPerCall


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

    def test_strip_keeps_prefill_spans_of_surviving_calls_and_drops_calls_beyond_the_cut(self, tokenizer):
        """A call whose output boundary no longer fits vanishes with its prefill spans; the survivor keeps its own."""
        s = _make_sample([1, 2], [3, 4, 5, 6])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)],
                prefill_spans=[WeightVersionSpan("v0", 0, 1), WeightVersionSpan("v1", 1, 2)],
                output_start=2,
            ),
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 4, 6)], prefill_spans=[WeightVersionSpan("v1", 0, 4)], output_start=4
            ),
        ]
        s.strip_last_output_tokens(3, tokenizer)
        assert s.weight_versions == [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 3)],
                prefill_spans=[WeightVersionSpan("v0", 0, 1), WeightVersionSpan("v1", 1, 2)],
                output_start=2,
            )
        ]
        s.validate()

    def test_strip_up_to_a_call_output_start_keeps_that_call_as_an_unstamped_turn(self, tokenizer):
        """Cutting exactly at a call's output boundary leaves the call with no output spans but its prefill spans."""
        s = _make_sample([1, 2], [3, 4, 5, 6])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)], prefill_spans=[WeightVersionSpan("v0", 0, 2)], output_start=2
            ),
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 4, 6)], prefill_spans=[WeightVersionSpan("v1", 0, 4)], output_start=4
            ),
        ]
        s.strip_last_output_tokens(2, tokenizer)
        assert s.weight_versions == [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)], prefill_spans=[WeightVersionSpan("v0", 0, 2)], output_start=2
            ),
            WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("v1", 0, 4)], output_start=4),
        ]
        s.validate()

    def test_strip_through_several_calls_drops_the_vanished_ones_and_clips_the_last_survivor(self, tokenizer):
        """Stripping through call k removes it entirely while call k-1 keeps its prefill spans and clipped output."""
        s = _make_sample([1, 2], [3, 4, 5, 6, 7, 8])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)], prefill_spans=[WeightVersionSpan("v0", 0, 2)], output_start=2
            ),
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 4, 6)], prefill_spans=[WeightVersionSpan("v1", 0, 4)], output_start=4
            ),
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v3", 6, 8)], prefill_spans=[WeightVersionSpan("v2", 0, 6)], output_start=6
            ),
        ]
        s.strip_last_output_tokens(3, tokenizer)
        assert s.weight_versions == [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)], prefill_spans=[WeightVersionSpan("v0", 0, 2)], output_start=2
            ),
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 4, 5)], prefill_spans=[WeightVersionSpan("v1", 0, 4)], output_start=4
            ),
        ]
        s.validate()

    def test_strip_drops_an_unstamped_call_once_its_output_boundary_is_past_the_cut(self, tokenizer):
        """A call without output spans is anchored by its own output boundary, not by its prefill spans."""
        s = _make_sample([1, 2], [3, 4, 5, 6])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)], prefill_spans=[WeightVersionSpan("v0", 0, 2)], output_start=2
            ),
            WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("v1", 0, 5)], output_start=5),
        ]
        s.strip_last_output_tokens(1, tokenizer)
        assert s.weight_versions == [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)], prefill_spans=[WeightVersionSpan("v0", 0, 2)], output_start=2
            ),
            WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("v1", 0, 5)], output_start=5),
        ]
        s.validate()
        s.strip_last_output_tokens(1, tokenizer)
        assert s.weight_versions == [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v1", 2, 4)], prefill_spans=[WeightVersionSpan("v0", 0, 2)], output_start=2
            )
        ]
        s.validate()

    def test_strip_keeps_calls_whose_output_boundary_is_unknown(self, tokenizer):
        """Legacy calls without an output boundary are clipped like before instead of being dropped."""
        s = _make_sample([1, 2], [3, 4, 5, 6])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 6)]),
        ]
        s.strip_last_output_tokens(3, tokenizer)
        assert s.weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 3)]),
            WeightVersionsPerCall(spans=[]),
        ]
        s.validate()


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
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4), WeightVersionSpan("v2", 4, 5)], output_start=2)
        ]

    def test_update_from_meta_info_synthesizes_span_from_scalar_weight_version(self):
        """Without per-token data, the scalar weight_version becomes one span over the new tokens."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4, 5], weight_version="v7"))
        assert s.weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("v7", 2, 5)], output_start=2)]

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

    def test_from_meta_info_rejects_span_beyond_reported_output_tokens(self):
        """A span reaching past the reported output tokens means the logprobs are incomplete, so it raises."""
        with pytest.raises(AssertionError, match="extend past the 2 output tokens"):
            WeightVersionsPerCall.from_meta_info(
                {
                    "output_token_logprobs": [(-0.1, 3), (-0.1, 4)],
                    "weight_versions": [{"version": "v1", "start": 0, "end": 3}],
                },
                output_end=5,
            )

    def test_per_token_weight_versions_take_precedence_over_scalar_weight_version(self):
        """When the engine reports both, the per-token spans win over the scalar fallback."""
        call = WeightVersionsPerCall.from_meta_info(
            {
                "output_token_logprobs": [(-0.1, 3), (-0.1, 4)],
                "weight_versions": [{"version": "v1", "start": 0, "end": 1}],
                "weight_version": "v9",
            },
            output_end=2,
        )
        assert call.spans == [WeightVersionSpan("v1", 0, 1)]

    def test_update_from_meta_info_records_a_call_without_weight_version(self):
        """A call the engine did not stamp still counts as one call, with no spans."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4, 5]))
        assert s.weight_versions == [WeightVersionsPerCall(spans=[], output_start=2)]
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
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)], output_start=2),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 7)], output_start=4),
        ]
        assert s.all_weight_version_spans == [WeightVersionSpan("v1", 2, 4), WeightVersionSpan("v2", 4, 7)]

    def test_reset_for_retry_clears_weight_versions(self):
        """reset_for_retry clears weight_versions along with other outputs."""
        s = _make_sample([1, 2], [3, 4])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)])]
        s.reset_for_retry()
        assert s.weight_versions == []

    def test_reset_for_retry_clears_the_policy_the_last_generation_stamped(self):
        """The policy id marks what a generation produced, so a retry must earn it again instead of inheriting it."""
        s = _make_sample([1, 2], [3, 4])
        s.trainer_model_id = "solver"

        s.reset_for_retry()

        assert s.trainer_model_id is None

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

    def test_validate_rejects_empty_or_reversed_weight_version_span(self):
        """validate fails for a zero-length span and for one whose end precedes its start."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 2)])]
        with pytest.raises(AssertionError, match="invalid weight version span"):
            s.validate()

        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 4, 3)])]
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

    def test_from_dict_keeps_pre_span_dumps_loadable(self):
        """Dumps predating per-call spans load without misparsing their flat version strings."""
        restored = Sample.from_dict({"status": "completed", "weight_versions": ["v1", "v2"], "tokens": [1, 2]})
        assert restored.weight_versions == []
        assert getattr(restored, LEGACY_WEIGHT_VERSIONS_KEY) == ["v1", "v2"]

    def test_from_dict_reads_current_dumps_unchanged(self):
        """A dump written with per-call span mappings round-trips into the typed structure."""
        restored = Sample.from_dict(
            {
                "status": "completed",
                "weight_versions": [
                    {
                        "spans": [{"version": "v1", "abs_start": 0, "abs_end": 2}],
                        "prefill_spans": [],
                        "output_start": 0,
                    }
                ],
                "tokens": [1, 2],
            }
        )
        assert restored.weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 0, 2)], output_start=0)
        ]
        assert not hasattr(restored, LEGACY_WEIGHT_VERSIONS_KEY)

    def test_validate_accepts_prefill_spans_covering_the_prompt(self):
        """validate passes when a call's prefill spans tile [0, output_start) in order."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 2, 5)],
                prefill_spans=[WeightVersionSpan("v1", 0, 1), WeightVersionSpan("v2", 1, 2)],
                output_start=2,
            )
        ]
        s.validate()

    def test_validate_rejects_output_spans_that_do_not_start_at_the_call_output_start(self):
        """validate fails when a call's first output span disagrees with its recorded output boundary."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 3, 5)], output_start=2)]
        with pytest.raises(AssertionError, match="must be non-empty and start at token 2"):
            s.validate()

    def test_validate_rejects_a_gap_between_output_spans(self) -> None:
        """validate rejects unstamped tokens between output spans of the same call."""
        sample = _make_sample([1, 2], [3, 4, 5])
        sample.weight_versions = [
            WeightVersionsPerCall(
                spans=[
                    WeightVersionSpan(version="v1", abs_start=2, abs_end=3),
                    WeightVersionSpan(version="v2", abs_start=4, abs_end=5),
                ],
                output_start=2,
            )
        ]

        with pytest.raises(AssertionError, match="must be non-empty and start at token 3"):
            sample.validate()

    def test_validate_rejects_an_output_start_past_the_tokens(self):
        """validate fails when a call claims an output boundary the sample's tokens never reach."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[], output_start=6)]
        with pytest.raises(AssertionError, match="starts its output at 6 but the sample has 5 tokens"):
            s.validate()

    def test_validate_rejects_prefill_spans_without_an_output_start(self):
        """Prefill spans are only meaningful against a known output boundary."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("v1", 0, 2)])]
        with pytest.raises(AssertionError, match="carries prefill spans without output_start"):
            s.validate()

    @pytest.mark.parametrize("prefill_end", [1, 3])
    def test_validate_rejects_prefill_spans_not_ending_at_the_call_output_start(self, prefill_end: int):
        """validate fails when a call's prefill spans stop short of or reach into its own output tokens."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 2, 5)],
                prefill_spans=[WeightVersionSpan("v1", 0, prefill_end)],
                output_start=2,
            )
        ]
        with pytest.raises(AssertionError, match="must cover exactly the 2 prompt tokens"):
            s.validate()

    def test_validate_rejects_prefill_spans_not_starting_at_the_first_token(self):
        """validate fails when the first prefill span of a call leaves the prompt head unstamped."""
        s = _make_sample([1, 2, 3], [4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 3, 5)], prefill_spans=[WeightVersionSpan("v1", 1, 3)], output_start=3
            )
        ]
        with pytest.raises(AssertionError, match="must be non-empty and start at token"):
            s.validate()

    def test_validate_rejects_a_gap_between_prefill_spans(self):
        """validate fails when two prefill spans of one call leave prompt tokens between them unstamped."""
        s = _make_sample([1, 2, 3, 4], [5, 6])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 4, 6)],
                prefill_spans=[WeightVersionSpan("v1", 0, 1), WeightVersionSpan("v2", 2, 4)],
                output_start=4,
            )
        ]
        with pytest.raises(AssertionError, match="must be non-empty and start at token"):
            s.validate()

    def test_validate_rejects_overlapping_prefill_spans(self):
        """validate fails when two prefill spans of one call overlap."""
        s = _make_sample([1, 2, 3], [4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 3, 5)],
                prefill_spans=[WeightVersionSpan("v1", 0, 2), WeightVersionSpan("v2", 1, 3)],
                output_start=3,
            )
        ]
        with pytest.raises(AssertionError, match="must be non-empty and start at token"):
            s.validate()

    def test_validate_rejects_empty_or_reversed_prefill_span(self):
        """validate fails for a zero-length prefill span and for one whose end precedes its start."""
        s = _make_sample([1, 2, 3], [4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 3, 5)], prefill_spans=[WeightVersionSpan("v1", 1, 1)], output_start=3
            )
        ]
        with pytest.raises(AssertionError, match="must be non-empty and start at token"):
            s.validate()

        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 3, 5)], prefill_spans=[WeightVersionSpan("v1", 2, 1)], output_start=3
            )
        ]
        with pytest.raises(AssertionError, match="must be non-empty and start at token"):
            s.validate()

    def test_validate_rejects_empty_prefill_version(self):
        """validate fails when a prefill span carries an empty version string."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("v2", 2, 5)], prefill_spans=[WeightVersionSpan("", 0, 2)], output_start=2
            )
        ]
        with pytest.raises(AssertionError, match="empty version"):
            s.validate()

    def test_validate_checks_the_prefill_spans_of_an_unstamped_call_against_its_output_start(self):
        """A call without output spans still needs its prefill spans to tile up to its recorded output boundary."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("v1", 0, 5)], output_start=5)
        ]
        s.validate()

        s.weight_versions = [
            WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("v1", 0, 2)], output_start=5)
        ]
        with pytest.raises(AssertionError, match="must cover exactly the 5 prompt tokens"):
            s.validate()

        s.weight_versions = [
            WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("v1", 0, 6)], output_start=6)
        ]
        with pytest.raises(AssertionError, match="starts its output at 6 but the sample has 5 tokens"):
            s.validate()

    def test_oldest_weight_version_reads_all_spans(self):
        """oldest_weight_version takes the minimum numeric version across every span."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("7", 2, 4), WeightVersionSpan("5", 4, 5)])]
        assert s.oldest_weight_version == 5

    def test_oldest_weight_version_ignores_nonnumeric_spans(self):
        """Nonnumeric version labels are skipped, and a sample carrying only those reports no version."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4), WeightVersionSpan("9", 4, 5)])
        ]
        assert s.oldest_weight_version == 9

        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)])]
        assert s.oldest_weight_version is None

    def test_oldest_weight_version_ignores_prefill_spans(self):
        """Prompt KV versions stay out of oldest_weight_version so staleness filtering keeps its meaning."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("7", 2, 5)], prefill_spans=[WeightVersionSpan("1", 0, 2)], output_start=2
            )
        ]
        s.validate()
        assert s.oldest_weight_version == 7

    def test_to_dict_from_dict_roundtrip_keeps_prefill_spans(self):
        """Prefill spans survive a Sample to_dict/from_dict roundtrip next to the output spans."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("2", 2, 4)],
                prefill_spans=[WeightVersionSpan("1", 0, 1), WeightVersionSpan("2", 1, 2)],
                output_start=2,
            ),
            WeightVersionsPerCall(spans=[WeightVersionSpan("3", 4, 5)], output_start=4),
        ]
        s.validate()
        restored = Sample.from_dict(s.to_dict())
        assert restored.weight_versions == s.weight_versions


class TestWeightVersionsPerCallFromMetaInfo:
    @pytest.mark.parametrize("prompt_tokens", [36, 68], ids=["image", "audio"])
    def test_expanded_media_prefill_keeps_engine_coordinates_through_roundtrip_and_trimming(
        self, prompt_tokens: int, tokenizer: MagicMock
    ) -> None:
        """Media expansion changes the prefill boundary without shifting sample output spans."""
        sample = _make_sample([1, 2, 3, 4, 5], [10, 11])
        sample.weight_versions = [
            WeightVersionsPerCall.from_meta_info(
                meta_info={
                    "prompt_tokens": prompt_tokens,
                    "output_token_logprobs": [(-0.1, 10), (-0.1, 11)],
                    "weight_versions": [{"version": "3", "start": 0, "end": 2}],
                    "prefill_weight_versions": [{"version": "2", "start": 0, "end": prompt_tokens}],
                },
                output_end=7,
            )
        ]
        sample.validate()

        restored = Sample.from_dict(sample.to_dict())
        restored.strip_last_output_tokens(n=1, tokenizer=tokenizer)
        restored.validate()

        assert restored.weight_versions == [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan("3", 5, 6)],
                prefill_spans=[WeightVersionSpan("2", 0, prompt_tokens)],
                output_start=5,
                prompt_tokens=prompt_tokens,
            )
        ]

    def test_reported_engine_prompt_length_rejects_incomplete_prefill(self) -> None:
        """A sample-sized prefill span cannot substitute for the expanded engine prompt."""
        call = WeightVersionsPerCall.from_meta_info(
            meta_info={
                "prompt_tokens": 36,
                "output_token_logprobs": [(-0.1, 10)],
                "prefill_weight_versions": [{"version": "2", "start": 0, "end": 5}],
            },
            output_end=6,
        )

        with pytest.raises(AssertionError, match="must cover exactly the 36 prompt tokens"):
            call.validate(num_tokens=6)

    def test_prefill_weight_versions_land_as_absolute_prompt_spans(self):
        """Prefill spans index the call's input_ids, which start at token 0, so they are kept as-is."""
        call = WeightVersionsPerCall.from_meta_info(
            {
                "output_token_logprobs": [(-0.1, 10), (-0.1, 11)],
                "weight_versions": [{"version": "3", "start": 0, "end": 2}],
                "prefill_weight_versions": [
                    {"version": "1", "start": 0, "end": 2},
                    {"version": "3", "start": 2, "end": 4},
                ],
            },
            output_end=6,
        )
        assert call == WeightVersionsPerCall(
            spans=[WeightVersionSpan("3", 4, 6)],
            prefill_spans=[WeightVersionSpan("1", 0, 2), WeightVersionSpan("3", 2, 4)],
            output_start=4,
        )

    def test_without_prefill_weight_versions_the_call_has_no_prefill_spans(self):
        """An engine without the prefill flag reports nothing, so the call carries only output spans."""
        call = WeightVersionsPerCall.from_meta_info(
            {"output_token_logprobs": [(-0.1, 10)], "weight_versions": [{"version": "3", "start": 0, "end": 1}]},
            output_end=3,
        )
        assert call == WeightVersionsPerCall(spans=[WeightVersionSpan("3", 2, 3)], output_start=2)

    def test_an_empty_prefill_list_lands_as_a_call_without_prefill_spans(self):
        """A present-but-empty list is the engine's answer for a prompt with no tokens."""
        call = WeightVersionsPerCall.from_meta_info(
            {"output_token_logprobs": [(-0.1, 10)], "prefill_weight_versions": []}, output_end=1
        )
        assert call == WeightVersionsPerCall(spans=[], prefill_spans=[], output_start=0)

    def test_a_non_list_prefill_payload_is_rejected(self):
        """The field is a list of spans; any other shape is a wire error, not an absent field."""
        with pytest.raises(AssertionError, match="must be a list of spans"):
            WeightVersionsPerCall.from_meta_info(
                {"output_token_logprobs": [(-0.1, 10)], "prefill_weight_versions": {"version": "2"}}, output_end=4
            )

    def test_prefill_spans_are_recorded_even_when_the_output_is_unstamped(self):
        """A call whose output carries no version still keeps the prompt KV versions it was told."""
        call = WeightVersionsPerCall.from_meta_info(
            {
                "output_token_logprobs": [(-0.1, 10)],
                "prefill_weight_versions": [{"version": "2", "start": 0, "end": 3}],
            },
            output_end=4,
        )
        assert call == WeightVersionsPerCall(spans=[], prefill_spans=[WeightVersionSpan("2", 0, 3)], output_start=3)

    @pytest.mark.parametrize("last_end", [3, 5])
    def test_prefill_spans_not_ending_at_the_prompt_length_are_rejected(self, last_end: int):
        """The last prefill span must end exactly where this call's output starts."""
        call = WeightVersionsPerCall.from_meta_info(
            {
                "output_token_logprobs": [(-0.1, 10), (-0.1, 11)],
                "prefill_weight_versions": [{"version": "2", "start": 0, "end": last_end}],
            },
            output_end=6,
        )

        with pytest.raises(AssertionError, match="must cover exactly the 4 prompt tokens"):
            call.validate(num_tokens=6)

    @pytest.mark.parametrize(
        "raw_spans",
        [
            pytest.param([{"version": "1", "start": 0, "end": 0}, {"version": "2", "start": 0, "end": 2}], id="empty"),
            pytest.param([{"version": "1", "start": 1, "end": 2}], id="not-from-zero"),
            pytest.param(
                [{"version": "2", "start": 1, "end": 2}, {"version": "1", "start": 0, "end": 1}], id="reordered"
            ),
            pytest.param([{"version": "1", "start": 0, "end": 1}, {"version": "2", "start": 2, "end": 2}], id="gap"),
            pytest.param(
                [{"version": "1", "start": 0, "end": 2}, {"version": "2", "start": 1, "end": 2}], id="overlap"
            ),
        ],
    )
    def test_prefill_spans_that_do_not_tile_the_prompt_are_rejected(self, raw_spans: list[dict]):
        """Prefill spans must start at token 0 and follow each other without gaps, overlaps or empty spans."""
        call = WeightVersionsPerCall.from_meta_info(
            {"output_token_logprobs": [(-0.1, 10)], "prefill_weight_versions": raw_spans}, output_end=3
        )

        with pytest.raises(AssertionError, match="must be non-empty and start at token"):
            call.validate(num_tokens=3)


class TestWeightVersionsPerCallDict:
    def test_older_mapping_without_engine_prompt_length_keeps_text_validation(self) -> None:
        """Older call mappings retain the sample-coordinate prompt boundary fallback."""
        call = WeightVersionsPerCall.from_dict({
            "spans": [{"version": "3", "abs_start": 4, "abs_end": 6}],
            "prefill_spans": [{"version": "2", "abs_start": 0, "abs_end": 3}],
            "output_start": 4,
        })

        assert call.prompt_tokens is None
        with pytest.raises(AssertionError, match="must cover exactly the 4 prompt tokens"):
            call.validate(num_tokens=6)

    def test_to_dict_writes_spans_and_prefill_spans(self):
        """The serialized call is a mapping holding both span lists."""
        call = WeightVersionsPerCall(
            spans=[WeightVersionSpan("3", 4, 6)], prefill_spans=[WeightVersionSpan("1", 0, 4)], output_start=4
        )
        assert call.to_dict() == {
            "spans": [{"version": "3", "abs_start": 4, "abs_end": 6}],
            "prefill_spans": [{"version": "1", "abs_start": 0, "abs_end": 4}],
            "output_start": 4,
            "prompt_tokens": None,
        }

    def test_to_dict_from_dict_roundtrips_output_start(self):
        """The call's output boundary is persisted next to its spans."""
        call = WeightVersionsPerCall(spans=[WeightVersionSpan("3", 4, 6)], output_start=4)
        assert call.to_dict()["output_start"] == 4
        assert WeightVersionsPerCall.from_dict(call.to_dict()) == call

    def test_from_dict_roundtrips_to_dict(self):
        """A serialized call reads back as an equal typed object."""
        call = WeightVersionsPerCall(
            spans=[WeightVersionSpan("3", 4, 6)],
            prefill_spans=[WeightVersionSpan("1", 0, 2), WeightVersionSpan("3", 2, 4)],
            output_start=4,
        )
        assert WeightVersionsPerCall.from_dict(call.to_dict()) == call

    def test_from_dict_roundtrips_an_empty_call(self):
        """An unstamped call serializes to two empty lists and reads back empty."""
        assert WeightVersionsPerCall.from_dict(WeightVersionsPerCall().to_dict()) == WeightVersionsPerCall()

    def test_from_dict_requires_output_start_in_the_mapping_shape(self):
        """A mapping without the output boundary is malformed and fails instead of defaulting."""
        with pytest.raises(KeyError, match="output_start"):
            WeightVersionsPerCall.from_dict({"spans": [], "prefill_spans": []})

    def test_from_dict_requires_both_span_lists_in_the_mapping_shape(self):
        """A mapping missing one of the span lists is malformed and fails instead of defaulting."""
        with pytest.raises(KeyError, match="prefill_spans"):
            WeightVersionsPerCall.from_dict({"spans": []})
