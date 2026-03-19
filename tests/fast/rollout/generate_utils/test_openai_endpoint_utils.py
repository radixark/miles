"""Tests for compute_samples_from_openai_records and TITO multi-turn merge workflow.

Validates the contract between session records, sample construction,
and merge_samples — the core of the TITO (Token In Token Out) pipeline.
"""

from unittest.mock import MagicMock

import pytest

from miles.rollout.generate_utils.openai_endpoint_utils import compute_samples_from_openai_records
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.router.session.session_types import SessionRecord
from miles.utils.types import Sample

# ── helpers ──────────────────────────────────────────────────────────


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
    """Build a minimal session record mimicking SGLang's response format."""
    if output_log_probs is None:
        output_log_probs = [-0.1 * (i + 1) for i in range(len(output_token_ids))]

    logprobs_content = [
        {"token_id": tid, "logprob": lp, "token": f"t{tid}"}
        for tid, lp in zip(output_token_ids, output_log_probs, strict=True)
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, [record], tok)

        assert len(samples) == 1
        s = samples[0]
        assert s.tokens == [1, 2, 3, 10, 11]
        assert s.rollout_log_probs == [-0.5, -0.6]
        assert s.response_length == 2
        assert s.loss_mask == [1, 1]
        assert s.status == Sample.Status.COMPLETED

    def test_multiple_records_produce_multiple_samples(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, records, tok)

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10]
        assert samples[1].tokens == [1, 2, 10, 20, 30]

    def test_finish_reason_length_gives_truncated(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2],
            output_token_ids=[10],
            finish_reason="length",
        )
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, [record], tok)

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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, records, tok)
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, records, tok)
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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, records, tok)

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)


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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, records, tok)

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
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]
