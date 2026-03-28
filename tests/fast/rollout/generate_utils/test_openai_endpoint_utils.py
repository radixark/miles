"""Tests for compute_samples_from_openai_records and TITO multi-turn merge workflow.

Validates the contract between session records, sample construction,
and merge_samples — the core of the TITO (Token In Token Out) pipeline.
"""

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from miles.rollout.generate_utils.openai_endpoint_utils import compute_samples_from_openai_records
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.router.session.session_types import SessionRecord
from miles.utils.types import Sample

# ── helpers ──────────────────────────────────────────────────────────


def _mock_args():
    return Namespace(use_rollout_routing_replay=False)


def _mock_tokenizer():
    tok = MagicMock()
    tok.decode = lambda ids: "".join(f"[{i}]" for i in ids)
    return tok


def _make_input_sample(**overrides):
    defaults = dict(
        group_index=0,
        index=0,
        prompt="test prompt",
        tokens=[],
        response="",
        response_length=0,
        status=Sample.Status.PENDING,
        label="test",
        reward=1.0,
    )
    defaults.update(overrides)
    return Sample(**defaults)


def _make_record(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    output_log_probs: list[float] | None = None,
    finish_reason: str = "stop",
) -> SessionRecord:
    """Build a minimal session record mimicking SGLang's response format.

    Token IDs and logprobs are stored in meta_info.output_token_logprobs
    as (logprob, token_id) tuples, matching the real SGLang response.
    """
    if output_log_probs is None:
        output_log_probs = [-0.1 * (i + 1) for i in range(len(output_token_ids))]

    output_token_logprobs = [(lp, tid) for tid, lp in zip(output_token_ids, output_log_probs, strict=True)]
    logprobs_content = [
        {"logprob": lp, "token": f"t{tid}"} for tid, lp in zip(output_token_ids, output_log_probs, strict=True)
    ]
    return SessionRecord(
        timestamp=0.0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={"messages": [{"role": "user", "content": "hello"}]},
        response={
            "choices": [
                {
                    "prompt_token_ids": prompt_token_ids,
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": finish_reason,
                    "logprobs": {"content": logprobs_content},
                    "meta_info": {
                        "output_token_logprobs": output_token_logprobs,
                        "completion_tokens": len(output_token_ids),
                    },
                }
            ]
        },
    )


# ── test: compute_samples_from_openai_records ────────────────────────


class TestComputeSamplesFromRecords:
    def test_single_record_builds_correct_sample(self):
        args = _mock_args()
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.5, -0.6],
        )
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, [record], tok)

        assert len(samples) == 1
        s = samples[0]
        assert s.tokens == [1, 2, 3, 10, 11]
        assert s.rollout_log_probs == [-0.5, -0.6]
        assert s.response_length == 2
        assert s.loss_mask == [1, 1]
        assert s.status == Sample.Status.COMPLETED

    def test_multiple_records_produce_multiple_samples(self):
        args = _mock_args()
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, records, tok)

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10]
        assert samples[1].tokens == [1, 2, 10, 20, 30]

    def test_finish_reason_length_gives_truncated(self):
        args = _mock_args()
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2],
            output_token_ids=[10],
            finish_reason="length",
        )
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, [record], tok)

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
        args = _mock_args()
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert merged.response_length == 2 + 2 + 2  # resp1 + obs + resp2
        assert merged.loss_mask == [1, 1, 0, 0, 1, 1]
        assert merged.status == Sample.Status.COMPLETED

    def test_three_turn_merge_succeeds(self):
        """Three-turn conversation: prefix chain holds across all turns."""
        args = _mock_args()
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 10, 20, 30, 40, 50]
        assert merged.response_length == 1 + 1 + 1 + 1 + 1  # 3 responses + 2 obs

    def test_prefix_mismatch_raises(self):
        """When the prefix chain is broken, merge_samples must fail."""
        args = _mock_args()
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, records, tok)

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
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
        args = _mock_args()
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            args,
            input_sample,
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
        args = _mock_args()
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            args,
            input_sample,
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
        args = _mock_args()
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11, STOP]),
        ]
        accumulated = [1, 2, 3, 10, 11, STOP]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            args,
            input_sample,
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
        args = _mock_args()
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, STOP, 3, 4], output_token_ids=[20, STOP]),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            args,
            input_sample,
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
        args = _mock_args()
        tok = _mock_tokenizer()

        # Output has 2 trailing tokens that don't match, but max_trim_tokens=1
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3, 4], output_token_ids=[20]),
        ]
        accumulated = [1, 2, 10, 3, 4, 20]
        input_sample = _make_input_sample()

        with pytest.raises(AssertionError, match="trim_count 2 exceeds allowed=1"):
            compute_samples_from_openai_records(
                args,
                input_sample,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )

    def test_cursor_covers_entire_accumulated(self):
        """After processing all records, cursor must equal len(accumulated)."""
        args = _mock_args()
        tok = _mock_tokenizer()

        # accumulated is shorter than what records imply — cursor won't reach end
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3], output_token_ids=[20]),
        ]
        # Missing the last token — accumulated should be [1,2,10,3,20] but we give [1,2,10,3,20,99]
        accumulated = [1, 2, 10, 3, 20, 99]
        input_sample = _make_input_sample()

        with pytest.raises(AssertionError, match="cursor .* != len\\(accumulated_token_ids\\)"):
            compute_samples_from_openai_records(
                args,
                input_sample,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )


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
        args = _mock_args()
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, records, tok)

        # sample[0].tokens = [1,2,3] + thinking + [10,11] = [1,2,3, <think>,\n,42,43,\n,</think>,\n, 10,11]
        # sample[1].tokens = [1,2,3, 10,11, 20,21, 30,31]
        # sample[1] does NOT start with sample[0] — prefix chain broken
        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)

    def test_no_thinking_tokens_prefix_chain_holds(self):
        """When thinking is disabled, the same conversation merges fine."""
        args = _mock_args()
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(args, input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]
