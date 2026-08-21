"""Tests for compute_samples_from_openai_records and TITO multi-turn merge workflow.

Validates the contract between session records, sample construction,
and merge_samples — the core of the TITO (Token In Token Out) pipeline.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pybase64
import pytest

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.samples.merge import compute_samples_from_openai_records, merge_samples_with_addition_r3
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample

# ── helpers ──────────────────────────────────────────────────────────

_ARGS = SimpleNamespace(save_debug_trajectory_data=None, sglang_speculative_algorithm=None)
_ARGS_RECORDING = SimpleNamespace(
    save_debug_trajectory_data="/unused/{rollout_id}.jsonl", sglang_speculative_algorithm=None
)


def _mock_tokenizer():
    tok = MagicMock()
    tok.decode = lambda ids: "".join(f"[{i}]" for i in ids)
    return tok


def _make_record(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    output_log_probs: list[float] | None = None,
    finish_reason: str = "stop",
    cached_tokens: int | None = None,
    prompt_tokens: int | None = None,
    weight_version: str | None = None,
    routed_experts: str | None = None,
    routed_experts_start_len: int | None = None,
    sampling_masks: list[list[int]] | None = None,
    sampling_log_probs: list[float] | None = None,
) -> SessionRecord:
    """Build a minimal session record mimicking SGLang's response format.

    Token IDs and logprobs are stored in meta_info.output_token_logprobs
    as (logprob, token_id) tuples, matching the real SGLang response.
    `routed_experts` is the base64 int32 buffer exactly as SGLang returns it;
    with `routed_experts_start_len` it covers only rows [start, end), matching
    an addition-R3 request that persisted the offset.
    """
    if output_log_probs is None:
        output_log_probs = [-0.1 * (i + 1) for i in range(len(output_token_ids))]

    output_token_logprobs = [(lp, tid) for tid, lp in zip(output_token_ids, output_log_probs, strict=True)]
    logprobs_content = [
        {"logprob": lp, "token": f"t{tid}"} for tid, lp in zip(output_token_ids, output_log_probs, strict=True)
    ]
    meta_info = {
        "output_token_logprobs": output_token_logprobs,
        "completion_tokens": len(output_token_ids),
    }
    if cached_tokens is not None:
        meta_info["cached_tokens"] = cached_tokens
    if prompt_tokens is not None:
        meta_info["prompt_tokens"] = prompt_tokens
    if weight_version is not None:
        meta_info["weight_version"] = weight_version
    if routed_experts is not None:
        meta_info["routed_experts"] = routed_experts
    if sampling_masks is not None:
        meta_info["output_token_sampling_mask"] = sampling_masks
        meta_info["output_token_sampling_logprobs"] = sampling_log_probs
    request = {"messages": [{"role": "user", "content": "hello"}], "input_ids": prompt_token_ids}
    if routed_experts_start_len is not None:
        request["routed_experts_start_len"] = routed_experts_start_len
    if sampling_masks is not None:
        request["return_sampling_mask"] = True
    return SessionRecord(
        timestamp=0.0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request=request,
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": finish_reason,
                    "logprobs": {"content": logprobs_content},
                    "meta_info": meta_info,
                }
            ]
        },
    )


# ── test: compute_samples_from_openai_records ────────────────────────


class TestComputeSamplesFromRecords:
    def test_single_record_builds_correct_sample(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.5, -0.6],
        )

        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert len(samples) == 1
        s = samples[0]
        assert s.tokens == [1, 2, 3, 10, 11]
        assert s.rollout_log_probs == [-0.5, -0.6]
        assert s.response_length == 2
        assert s.loss_mask == [1, 1]
        assert s.status == Sample.Status.COMPLETED

    def test_single_record_uses_native_sampling_support_and_log_probs(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-2.5, -2.6],
            sampling_masks=[[10, 4, 7], [11, 3]],
            sampling_log_probs=[-0.5, -0.6],
        )

        (sample,) = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert sample.rollout_log_probs == [-0.5, -0.6]
        assert sample.rollout_sampling_mask_ids == [10, 4, 7, 11, 3]
        assert sample.rollout_sampling_mask_offsets == [0, 3, 5]
        sample.validate()

    def test_abort_without_sampling_metadata_is_non_trainable(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[],
            finish_reason="abort",
        )
        record.request["return_sampling_mask"] = True

        (sample,) = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert sample.status == Sample.Status.ABORTED
        assert sample.reward is None
        assert sample.rollout_sampling_mask_ids is None
        assert sample.rollout_sampling_mask_offsets is None

    def test_successful_turn_still_requires_sampling_metadata(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10],
            finish_reason="stop",
        )
        record.request["return_sampling_mask"] = True

        with pytest.raises(ValueError, match="missing output_token_sampling_mask"):
            compute_samples_from_openai_records(_ARGS, [record], tok)

    def test_later_abort_keeps_fully_replayable_prefix(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(
                prompt_token_ids=[1, 2],
                output_token_ids=[10],
                sampling_masks=[[10, 11]],
                sampling_log_probs=[-0.2],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 10, 20],
                output_token_ids=[],
                finish_reason="abort",
            ),
        ]
        records[1].request["return_sampling_mask"] = True

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged is samples[0]
        assert merged.status == Sample.Status.COMPLETED
        assert merged.tokens == [1, 2, 10]

    def test_multiple_records_produce_multiple_samples(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10]
        assert samples[1].tokens == [1, 2, 10, 20, 30]

    def test_last_sample_carries_raw_conversation(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]

        samples = compute_samples_from_openai_records(_ARGS_RECORDING, records, tok)

        assert "messages" not in (samples[0].metadata or {})
        assert samples[1].metadata["messages"] == records[1].request["messages"] + [
            records[1].response["choices"][0]["message"]
        ]

    def test_merge_keeps_last_conversation_snapshot(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]

        samples = compute_samples_from_openai_records(_ARGS_RECORDING, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.metadata["messages"] == samples[-1].metadata["messages"]

    def test_finish_reason_length_gives_truncated(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2],
            output_token_ids=[10],
            finish_reason="length",
        )

        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].status == Sample.Status.TRUNCATED


# ── test: multi-turn prefix chain (merge_samples integration) ────────


class TestMultiTurnPrefixChain:
    """Validate that session records from a well-behaved multi-turn
    conversation satisfy the prefix chain required by merge_samples.

    The contract: for consecutive records r[i] and r[i+1],
    r[i+1].prompt_token_ids must start with r[i].prompt_token_ids + r[i].output_token_ids.
    This is because the agent includes the previous response in the next prompt.
    """

    def test_two_turn_merge_succeeds(self):
        """Normal two-turn conversation: samples merge without error."""
        tok = _mock_tokenizer()

        # Turn 1: prompt=[1,2,3], model outputs [10,11]
        # Turn 2: prompt=[1,2,3, 10,11, 20,21], model outputs [30,31]
        #   (tokens 20,21 are the tool/observation tokens added by the environment)
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
                output_log_probs=[-0.1, -0.2],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert merged.response_length == 2 + 2 + 2  # resp1 + obs + resp2
        assert merged.loss_mask == [1, 1, 0, 0, 1, 1]
        assert merged.status == Sample.Status.COMPLETED

    def test_three_turn_merge_succeeds(self):
        """Three-turn conversation: prefix chain holds across all turns."""
        tok = _mock_tokenizer()

        records = [
            _make_record(
                prompt_token_ids=[1, 2],
                output_token_ids=[10],
                output_log_probs=[-0.1],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 10, 20],
                output_token_ids=[30],
                output_log_probs=[-0.2],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 10, 20, 30, 40],
                output_token_ids=[50],
                output_log_probs=[-0.3],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 10, 20, 30, 40, 50]
        assert merged.response_length == 1 + 1 + 1 + 1 + 1  # 3 responses + 2 obs

    def test_prefix_mismatch_raises(self):
        """When the prefix chain is broken, merge_samples must fail."""
        tok = _mock_tokenizer()

        # Turn 2's prompt does NOT start with turn 1's full tokens
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 99, 99, 20, 21],  # 99,99 != 10,11
                output_token_ids=[30, 31],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)

    def test_two_turn_merge_propagates_teacher_log_probs(self):
        """OPD teacher_log_probs merge like rollout_log_probs: per-turn values
        concatenated with zeros over the injected observation span."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11], output_log_probs=[-0.1, -0.2]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        # OPD attaches per-response-token teacher log-probs to each turn's sample.
        samples[0].teacher_log_probs = [-1.0, -1.1]
        samples[1].teacher_log_probs = [-1.2, -1.3]

        merged = merge_samples(samples, tok)

        # resp1 (2) + obs (2 zeros) + resp2 (2)
        assert merged.teacher_log_probs == [-1.0, -1.1, 0.0, 0.0, -1.2, -1.3]
        assert len(merged.teacher_log_probs) == merged.response_length
        merged.validate()  # the new teacher_log_probs length assertion must hold

    def test_two_turn_merge_propagates_opd_student_top_logprobs_metadata(self):
        """Top-k OPD student top-logprobs are per-token metadata, not equal metadata."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11], output_log_probs=[-0.1, -0.2]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        turn_0_top_logprobs = [[[-0.1, 101]], [[-0.2, 102]]]
        turn_1_top_logprobs = [[[-0.3, 103]], [[-0.4, 104]]]
        samples[0].metadata = {
            "opd_student_top_logprobs": turn_0_top_logprobs,
            "shared_metadata": "same",
        }
        samples[1].metadata = {
            "opd_student_top_logprobs": turn_1_top_logprobs,
            "shared_metadata": "same",
        }

        merged = merge_samples(samples, tok)

        assert merged.metadata["shared_metadata"] == "same"
        assert merged.metadata["opd_student_top_logprobs"] == [
            *turn_0_top_logprobs,
            [],
            [],
            *turn_1_top_logprobs,
        ]
        assert len(merged.metadata["opd_student_top_logprobs"]) == merged.response_length

    def test_two_turn_merge_teacher_log_probs_none_stays_none(self):
        """Non-OPD runs leave teacher_log_probs unset; merge must keep it None."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        merged = merge_samples(samples, tok)

        assert merged.teacher_log_probs is None

    def test_merge_raises_on_teacher_log_probs_length_mismatch(self):
        """validate() guards teacher_log_probs length (surfaced via merge_samples)."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        samples[0].teacher_log_probs = [-1.0]  # length 1 != response_length 2

        with pytest.raises(AssertionError, match="teacher_log_probs length"):
            merge_samples(samples, tok)


# ── test: TITO trailing token trimming ────────────────────────────────

STOP = 99  # stands for <|observation|> stop token


class TestTITOTrailingTokenTrim:
    """Validate trailing-token trimming via ``accumulated_token_ids``.

    Worked example — agentic tool-call retries
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    An agent makes three turns.  The model's tool call fails to parse on
    turns 1 and 2, so the agent feeds back an error and retries.

    The session server sees three request/response pairs (records).  Each
    record's response is an independent inference re-stitched via
    pretokenized prefix reuse::

        record 0  prompt_token_ids:  [<|sys|>, aaa, <|user|>, bbb, <|asst|>]
                  output_token_ids:  [ccc, <|obs|>]      ← model stopped with <|obs|>

        record 1  prompt_token_ids:  [<|sys|>, aaa, <|user|>, bbb, <|asst|>, ccc, <|sys|>, ddd, <|asst|>]
                  output_token_ids:  [eee, <|obs|>]

        record 2  prompt_token_ids:  [..., eee, <|sys|>, fff, <|asst|>]
                  output_token_ids:  [ggg, <|obs|>]

    ``accumulated_token_ids`` = record 2's prompt + output::

        [<|sys|>, aaa, <|user|>, bbb, <|asst|>, ccc, <|sys|>, ddd,
         <|asst|>, eee, <|sys|>, fff, <|asst|>, ggg, <|obs|>]

    Note: there is NO ``<|obs|>`` between ``ccc`` and ``<|sys|>`` in the
    accumulated sequence — the stop token the model emitted at turn 1 was
    consumed by the chat template when rendering turn 2's prompt.

    The algorithm walks ``accumulated_token_ids`` with a cursor::

        Record 0:  cursor = len(prompt_0) → points to "ccc"
                   Match output [ccc, <|obs|>] against accumulated[cursor:]:
                     ccc OK, <|obs|> MISMATCH (accumulated has <|sys|> here)
                   → trim_count=1, strip <|obs|>; cursor advances past "ccc"

        Record 1:  cursor = len(prompt_1) → points to "eee"
                   Match [eee, <|obs|>]: eee OK, <|obs|> MISMATCH
                   → trim_count=1; cursor advances past "eee"

        Record 2:  cursor = len(prompt_2) → points to "ggg"
                   Match [ggg, <|obs|>]: ggg OK, <|obs|> OK (last turn)
                   → trim_count=0; cursor reaches end

    Result: three Samples with output tokens [ccc], [eee], [ggg, <|obs|>],
    each carrying original per-turn logprobs.

    The tests below encode this example (and variants) with concrete
    token IDs.  We use ``STOP = 99`` to represent ``<|observation|>``.
    """

    def test_three_turn_trim_trailing_stop_tokens(self):
        """Three-turn retry: non-final turns have 1 trailing stop token trimmed."""
        tok = _mock_tokenizer()

        #   prompt: [1, 2, 3]  output: [10, STOP]
        #   prompt: [1, 2, 3, 10, 4, 5, 6]  output: [20, STOP]
        #   prompt: [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9]  output: [30, STOP]
        # accumulated (no intermediate STOPs):
        #   [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 4, 5, 6], output_token_ids=[20, STOP]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9], output_token_ids=[30, STOP]),
        ]
        accumulated = [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 3
        # Turn 0: [10, STOP] → trim 1 → response_length=1
        assert samples[0].tokens == [1, 2, 3, 10]
        assert samples[0].response_length == 1
        # Turn 1: [20, STOP] → trim 1 → response_length=1
        assert samples[1].tokens == [1, 2, 3, 10, 4, 5, 6, 20]
        assert samples[1].response_length == 1
        # Turn 2 (last): [30, STOP] → trim 0 → response_length=2
        assert samples[2].tokens == [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]
        assert samples[2].response_length == 2

    def test_no_trim_when_no_trailing_stop(self):
        """When output tokens fully match accumulated, trim_count=0 for all turns."""
        tok = _mock_tokenizer()

        # Two turns, no trailing stop tokens — output aligns perfectly
        #   prompt: [1, 2]  output: [10, 11]
        #   prompt: [1, 2, 10, 11, 3, 4]  output: [20, 21]
        # accumulated: [1, 2, 10, 11, 3, 4, 20, 21]
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 10, 11, 3, 4], output_token_ids=[20, 21]),
        ]
        accumulated = [1, 2, 10, 11, 3, 4, 20, 21]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10, 11]
        assert samples[0].response_length == 2
        assert samples[1].tokens == [1, 2, 10, 11, 3, 4, 20, 21]
        assert samples[1].response_length == 2

    def test_single_turn_no_trim(self):
        """Single turn: last turn never trims, even with accumulated_token_ids."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11, STOP]),
        ]
        accumulated = [1, 2, 3, 10, 11, STOP]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 1
        assert samples[0].tokens == [1, 2, 3, 10, 11, STOP]
        assert samples[0].response_length == 3

    def test_no_accumulated_skips_trimming(self):
        """Without accumulated_token_ids, no trimming is performed at all."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, STOP, 3, 4], output_token_ids=[20, STOP]),
        ]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=None,
        )

        assert len(samples) == 2
        # No trimming — STOP is kept for both turns
        assert samples[0].tokens == [1, 2, 10, STOP]
        assert samples[0].response_length == 2
        assert samples[1].tokens == [1, 2, 10, STOP, 3, 4, 20, STOP]
        assert samples[1].response_length == 2

    def test_trim_exceeding_max_raises(self):
        """If trailing tokens exceed max_trim_tokens, assert fires."""
        tok = _mock_tokenizer()

        # Output has 2 trailing tokens that don't match, but max_trim_tokens=1
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3, 4], output_token_ids=[20]),
        ]
        accumulated = [1, 2, 10, 3, 4, 20]

        with pytest.raises(AssertionError, match="trim_count 2 exceeds allowed=1"):
            compute_samples_from_openai_records(
                _ARGS,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )

    def test_cursor_covers_entire_accumulated(self):
        """After processing all records, cursor must equal len(accumulated)."""
        tok = _mock_tokenizer()

        # accumulated is shorter than what records imply — cursor won't reach end
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3], output_token_ids=[20]),
        ]
        # Missing the last token — accumulated should be [1,2,10,3,20] but we give [1,2,10,3,20,99]
        accumulated = [1, 2, 10, 3, 20, 99]

        with pytest.raises(AssertionError, match="cursor .* != len\\(accumulated_token_ids\\)"):
            compute_samples_from_openai_records(
                _ARGS,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )


# ── test: additional-R3 patch assembly (in-place weight updates) ──────

_ARGS_R3 = SimpleNamespace(save_debug_trajectory_data=None, sglang_speculative_algorithm=None, num_layers=2)
_R3_TOPK = 2


def _r3_rows(num_rows: int, seed: int) -> np.ndarray:
    """`num_rows` distinct rows of shape (num_layers, topk), values seed, seed+1, …"""
    size = num_rows * _ARGS_R3.num_layers * _R3_TOPK
    return np.arange(seed, seed + size, dtype=np.int32).reshape(num_rows, _ARGS_R3.num_layers, _R3_TOPK)


def _r3_patch(rows: np.ndarray) -> str:
    return pybase64.b64encode(np.ascontiguousarray(rows).tobytes()).decode("ascii")


def _merge_addition(records, accumulated, max_trim_tokens=0):
    tok = _mock_tokenizer()
    samples = compute_samples_from_openai_records(
        _ARGS_R3,
        records,
        tok,
        accumulated_token_ids=accumulated,
        max_trim_tokens=max_trim_tokens,
        use_addition_r3=True,
    )
    return merge_samples_with_addition_r3(_ARGS_R3, samples, records, tok)


class TestAdditionR3Assembly:
    """Addition-mode records carry an R3 patch of rows [routed_experts_start_len,
    len(input_ids) + len(output) - 1) instead of a full per-turn tensor;
    `merge_samples_with_addition_r3` concatenates the append-only stream into
    one (len(tokens) - 1, num_layers, topk) tensor after the ordinary merge.
    """

    def test_two_patches_reconstruct_full_reference(self):
        """Patches must rebuild byte-for-byte the tensor full-R3 responses give
        for the same tokens under the same in-place lifecycle."""
        full = _r3_rows(7, seed=0)
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21]
        # Turn 1: 5-token checkpoint -> rows [0, 4). Turn 2 extends it by one
        # env token -> start = checkpoint rows = 4, rows [4, 7).
        addition_records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(full[:4]), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4], [20, 21], routed_experts=_r3_patch(full[4:]), routed_experts_start_len=4
            ),
        ]
        full_records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(full[:4])),
            _make_record([1, 2, 3, 10, 11, 4], [20, 21], routed_experts=_r3_patch(full)),
        ]

        merged_addition = _merge_addition(addition_records, accumulated)
        tok = _mock_tokenizer()
        full_samples = compute_samples_from_openai_records(
            _ARGS_R3, full_records, tok, accumulated_token_ids=accumulated
        )
        merged_full = merge_samples(full_samples, tok)

        assert merged_addition.tokens == merged_full.tokens == accumulated
        assert merged_addition.loss_mask == merged_full.loss_mask
        assert merged_addition.rollout_log_probs == merged_full.rollout_log_probs
        assert merged_addition.rollout_routed_experts.dtype == np.int32
        assert np.array_equal(merged_addition.rollout_routed_experts, merged_full.rollout_routed_experts)
        assert np.array_equal(merged_addition.rollout_routed_experts, full)

    def test_overlapping_start_raises(self):
        """Every patch must start exactly after the preceding raw patch."""
        turn1_rows = _r3_rows(5, seed=0)
        turn2_rows = _r3_rows(2, seed=500)
        records = [
            _make_record([1, 2, 3], [10, 98, 99], routed_experts=_r3_patch(turn1_rows), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 55], [20, 21], routed_experts=_r3_patch(turn2_rows), routed_experts_start_len=4
            ),
        ]
        accumulated = [1, 2, 3, 10, 55, 20, 21]

        with pytest.raises(ValueError, match="record 1 starts at row 4; expected 5"):
            _merge_addition(records, accumulated, max_trim_tokens=2)

    def test_intermediate_truncated_turn_bounds_materialization(self):
        """A non-COMPLETED turn bounds required rows before later patches."""
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4],
                [20, 21],
                finish_reason="length",
                routed_experts=_r3_patch(_r3_rows(3, 16)),
                routed_experts_start_len=4,
            ),
            _make_record([1, 2, 3, 10, 11, 4, 20, 21, 5], [30, 31]),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21, 5, 30, 31]

        merged = _merge_addition(records, accumulated)

        assert merged.status == Sample.Status.TRUNCATED
        assert merged.tokens == accumulated[:8]
        expected = np.concatenate([_r3_rows(4, 0), _r3_rows(3, 16)])
        assert np.array_equal(merged.rollout_routed_experts, expected)

    def test_missing_required_patch_raises(self):
        """A missing patch cannot silently shorten a merged trajectory."""
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4],
                [20, 21],
                routed_experts=_r3_patch(_r3_rows(3, 16)),
                routed_experts_start_len=4,
            ),
            _make_record([1, 2, 3, 10, 11, 4, 20, 21, 5], [30, 31]),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21, 5, 30, 31]

        with pytest.raises(ValueError, match="record 2 has no routed_experts payload"):
            _merge_addition(records, accumulated)

    def test_replay_disabled_records_unaffected(self):
        """use_addition_r3=True is dormant when no record carries R3."""
        records = [
            _make_record([1, 2, 3], [10, 11]),
            _make_record([1, 2, 3, 10, 11, 4], [20, 21]),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21]

        merged = _merge_addition(records, accumulated)
        tok = _mock_tokenizer()
        reference = merge_samples(
            compute_samples_from_openai_records(_ARGS_R3, records, tok, accumulated_token_ids=accumulated), tok
        )

        assert merged.rollout_routed_experts is None
        assert merged.tokens == reference.tokens
        assert merged.loss_mask == reference.loss_mask

    def test_empty_delta_keeps_empty_buffer_shape(self):
        """A trajectory whose only patch is empty keeps the existing
        empty-buffer decode shape instead of inventing a top-k source."""
        records = [_make_record([1], [], routed_experts="", routed_experts_start_len=0)]

        merged = _merge_addition(records, [1])

        assert merged.rollout_routed_experts.shape == (0, _ARGS_R3.num_layers, 0)
        assert merged.rollout_routed_experts.dtype == np.int32

    def test_missing_start_len_raises(self):
        records = [_make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)))]

        with pytest.raises(ValueError, match="carries no routed_experts_start_len"):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_empty_payload_for_nonempty_delta_raises(self):
        records = [_make_record([1, 2, 3], [10, 11], routed_experts="", routed_experts_start_len=0)]

        with pytest.raises(ValueError, match="payload presence does not match 4 rows"):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_gapped_start_raises(self):
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(3, 0)), routed_experts_start_len=1)
        ]

        with pytest.raises(ValueError, match="starts at row 1; expected 0"):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_wrong_value_count_raises(self):
        # 4 rows announced (start=0, end=4) but only 3 rows of values supplied.
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(3, 0)), routed_experts_start_len=0)
        ]

        with pytest.raises(ValueError):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_inconsistent_topk_raises(self):
        # Turn 1 infers topk=2; turn 2 supplies 3 rows sized for topk=3.
        bad_rows = np.arange(3 * _ARGS_R3.num_layers * 3, dtype=np.int32).reshape(3, _ARGS_R3.num_layers, 3)
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4], [20, 21], routed_experts=_r3_patch(bad_rows), routed_experts_start_len=4
            ),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21]

        with pytest.raises(ValueError):
            _merge_addition(records, accumulated)


# ── test: thinking token issue (documents known failure mode) ────────


class TestThinkingTokenPrefixBreak:
    """Documents the known issue where model-generated <think>...</think>
    tokens break the prefix chain.

    When a model (e.g. Qwen3) generates <think>reasoning</think> before
    the actual response, agents strip the thinking content from conversation
    history. This causes the next turn's prompt to not include the thinking
    tokens, breaking the prefix assumption in merge_samples.

    This is a MODEL-LEVEL issue — the fix should be at the model/serving
    config level (disable thinking mode), not in the merge logic.
    """

    THINK_TOKEN = 151667  # <think> in Qwen3
    END_THINK_TOKEN = 151668  # </think> in Qwen3
    NEWLINE_TOKEN = 198  # \n

    def test_thinking_tokens_break_prefix_chain(self):
        """Demonstrates the failure: model outputs <think>..., but the agent
        strips it from history, so the next prompt doesn't include those tokens."""
        tok = _mock_tokenizer()

        # Turn 1: model generates <think>\nreasoning\n</think>\n then actual response
        thinking_tokens = [
            self.THINK_TOKEN,
            self.NEWLINE_TOKEN,
            42,
            43,
            self.NEWLINE_TOKEN,
            self.END_THINK_TOKEN,
            self.NEWLINE_TOKEN,
        ]
        response_tokens = [10, 11]
        all_output = thinking_tokens + response_tokens

        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=all_output,
            ),
            # Turn 2: agent only included the actual response [10, 11] in history
            # (stripped thinking tokens), plus observation [20, 21]
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        # sample[0].tokens = [1,2,3] + thinking + [10,11] = [1,2,3, <think>,\n,42,43,\n,</think>,\n, 10,11]
        # sample[1].tokens = [1,2,3, 10,11, 20,21, 30,31]
        # sample[1] does NOT start with sample[0] — prefix chain broken
        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)

    def test_no_thinking_tokens_prefix_chain_holds(self):
        """When thinking is disabled, the same conversation merges fine."""
        tok = _mock_tokenizer()

        # Same conversation but model output has no thinking prefix
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]


# ── test: prefix cache info population ────────────────────────────────


class TestPrefixCacheInfo:
    """Validate that prefix cache statistics from meta_info are collected."""

    def test_single_record_with_cache_stats(self):
        """cached_tokens and prompt_tokens from meta_info populate prefix_cache_info."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            cached_tokens=2,
            prompt_tokens=3,
        )
        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].prefix_cache_info.cached_tokens == 2
        assert samples[0].prefix_cache_info.total_prompt_tokens == 3

    def test_multi_turn_cache_stats_accumulate_after_merge(self):
        """After merge_samples, prefix_cache_info sums across turns."""
        tok = _mock_tokenizer()
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
                output_log_probs=[-0.1, -0.2],
                cached_tokens=0,
                prompt_tokens=3,
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
                cached_tokens=5,
                prompt_tokens=7,
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.prefix_cache_info.cached_tokens == 0 + 5
        assert merged.prefix_cache_info.total_prompt_tokens == 3 + 7
        assert merged.prefix_cache_info.prefix_cache_hit_rate == 5 / 10

    def test_missing_cache_fields_default_to_zero(self):
        """Records without cached_tokens/prompt_tokens give zero prefix_cache_info (regression)."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
        )
        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].prefix_cache_info.cached_tokens == 0
        assert samples[0].prefix_cache_info.total_prompt_tokens == 0
