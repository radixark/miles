"""Tests for TokenSeqComparator: segmentation, comparison, and mismatch classification.

## Background

TokenSeqComparator is used to verify TITO correctness.  After the TITO
pipeline merges pretokenized prefix tokens with incremental tokens, the
comparator checks the merged result against the full template-rendered
sequence.  It segments both by special-token boundaries, then classifies
any differences as structural, assistant-content, or non-assistant-content
mismatches.

## Test structure

TestSegmentation
    Verifies segment_by_special_tokens correctly splits token ID sequences
    at special-token boundaries.  This is the foundation for all comparison
    logic — if segmentation is wrong, mismatch classification is meaningless.

TestCompareIdentical
    Baseline: identical sequences and empty sequences produce no mismatches.

TestStructuralMismatch
    Tests for mismatches in the *structure* of the sequence (different number
    of segments, or a special token swapped for another).  These are the most
    severe mismatches — they mean the TITO merge produced a fundamentally
    different template structure.

TestContentMismatch
    Tests for mismatches in *content* between special tokens.  The comparator
    classifies these differently depending on whether the content belongs to
    an assistant turn or not:
    - ASSISTANT_TEXT: expected and benign — the model's generated text won't
      match the template's canonical re-tokenization (different sampling).
    - NON_ASSISTANT_TEXT: unexpected — tool responses, system prompts, and
      user messages should match exactly after TITO merge.

    Classification depends on the model's template structure:
    - Qwen3: ``<|im_start|>assistant\\n...`` → content starting with
      "assistant" after ``<|im_start|>`` is classified as assistant.
    - GLM47: ``<|assistant|>...`` → content after the ``<|assistant|>``
      special token is classified as assistant.

TestTrimTrailing
    Tests for the trim_trailing_ids feature, which strips specified token IDs
    from both sequence tails before comparison.  This handles model stop tokens
    (e.g. ``<|observation|>`` for GLM, ``<|im_end|>`` + ``\\n`` for Qwen)
    that appear at the end of generated output but not in the template-rendered
    expected sequence.  Without trimming, these would cause false structural
    mismatches.

TestGlm47BoundaryTokens
    GLM-specific regression test: ``<|user|>`` and ``<|observation|>`` are
    both valid next-turn openers after an assistant tool call, but swapping
    one for the other must be detected as a SPECIAL_TOKEN_TYPE mismatch.

## Test matrix

All tests using the ``env`` fixture run against both Qwen3-4B and GLM-4.7-Flash.
Model-specific tests (qwen3_env / glm47_env) test classification logic that
depends on the template's assistant-start pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.token_seq_comparator import MismatchType, Segment, TokenSeqComparator

# ---------------------------------------------------------------------------
# Model configs & fixtures
#
# Each ModelConfig captures the model-specific parameters needed to construct
# a TokenSeqComparator: the assistant_start_str (used for assistant vs
# non-assistant classification) and known special tokens (used to build
# test sequences with realistic token IDs).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    assistant_start_str: str
    known_special_tokens: tuple[str, ...]


_CONFIGS: dict[str, ModelConfig] = {
    "qwen3_4b": ModelConfig(
        model_id="Qwen/Qwen3-4B",
        assistant_start_str="<|im_start|>assistant",
        known_special_tokens=("<|im_start|>", "<|im_end|>", "<|endoftext|>"),
    ),
    "glm47_flash": ModelConfig(
        model_id="zai-org/GLM-4.7-Flash",
        assistant_start_str="<|assistant|>",
        known_special_tokens=("<|assistant|>", "<|user|>", "<|system|>", "<|observation|>", "<|endoftext|>"),
    ),
}


@dataclass
class TokenizerEnv:
    """Bundle of tokenizer + comparator + helpers for concise test code."""

    tokenizer: AutoTokenizer
    config: ModelConfig
    comparator: TokenSeqComparator

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_id(self, token_text: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token_text)


_ENV_CACHE: dict[str, TokenizerEnv] = {}


def _build_env(cfg: ModelConfig) -> TokenizerEnv:
    if cfg.model_id in _ENV_CACHE:
        return _ENV_CACHE[cfg.model_id]
    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    comp = TokenSeqComparator(tok, assistant_start_str=cfg.assistant_start_str)
    env = TokenizerEnv(tokenizer=tok, config=cfg, comparator=comp)
    _ENV_CACHE[cfg.model_id] = env
    return env


@pytest.fixture(params=list(_CONFIGS.keys()))
def env(request) -> TokenizerEnv:
    """Parametrized: runs the test once per model (Qwen3, GLM47)."""
    return _build_env(_CONFIGS[request.param])


@pytest.fixture
def qwen3_env() -> TokenizerEnv:
    return _build_env(_CONFIGS["qwen3_4b"])


@pytest.fixture
def glm47_env() -> TokenizerEnv:
    return _build_env(_CONFIGS["glm47_flash"])


# ---------------------------------------------------------------------------
# TestSegmentation — segment_by_special_tokens
#
# The comparator first segments a flat token ID list into alternating
# "special" and "content" segments.  Each special token gets its own
# single-ID segment; consecutive non-special tokens are grouped together.
#
# Example (Qwen3):
#   [<|im_start|>, 'assistant', '\n', 'Hi', <|im_end|>]
#   → [special(<|im_start|>), content('assistant\nHi'), special(<|im_end|>)]
# ---------------------------------------------------------------------------


class TestSegmentation:
    def test_empty(self, env: TokenizerEnv):
        assert env.comparator.segment_by_special_tokens([]) == []

    def test_plain_text_single_segment(self, env: TokenizerEnv):
        """No special tokens → one content segment containing all IDs."""
        ids = env.encode("The quick brown fox.")
        segs = env.comparator.segment_by_special_tokens(ids)
        assert len(segs) == 1
        assert segs[0].is_special is False
        assert segs[0].token_ids == ids

    def test_special_tokens_create_boundaries(self, env: TokenizerEnv):
        """<sp1> text <sp2> → [special, content, special]."""
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        text_ids = env.encode("some content")
        segs = env.comparator.segment_by_special_tokens([sp1] + text_ids + [sp2])
        assert len(segs) == 3
        assert segs[0] == Segment(token_ids=[sp1], is_special=True)
        assert segs[1] == Segment(token_ids=text_ids, is_special=False)
        assert segs[2] == Segment(token_ids=[sp2], is_special=True)

    def test_consecutive_specials_each_get_own_segment(self, env: TokenizerEnv):
        """Adjacent special tokens are NOT merged — each gets its own segment."""
        sp_ids = [env.token_id(t) for t in env.config.known_special_tokens[:3]]
        segs = env.comparator.segment_by_special_tokens(sp_ids)
        assert len(segs) == len(sp_ids)
        assert all(s.is_special and len(s.token_ids) == 1 for s in segs)


# ---------------------------------------------------------------------------
# TestCompareIdentical — baseline: no mismatches when sequences match
# ---------------------------------------------------------------------------


class TestCompareIdentical:
    def test_identical(self, env: TokenizerEnv):
        sp = env.token_id(env.config.known_special_tokens[0])
        ids = [sp] + env.encode("Hello world") + [sp]
        assert env.comparator.compare_sequences(ids, ids) == []

    def test_both_empty(self, env: TokenizerEnv):
        assert env.comparator.compare_sequences([], []) == []


# ---------------------------------------------------------------------------
# TestStructuralMismatch — segment count or special token identity differs
#
# These are the most severe mismatches: the template structure itself is
# wrong (e.g. a turn delimiter is missing or swapped).  In production,
# this means the TITO merge produced a prompt with incorrect turn boundaries.
# ---------------------------------------------------------------------------


class TestStructuralMismatch:
    def test_different_segment_count(self, env: TokenizerEnv):
        """Missing a special token → different segment count → SPECIAL_TOKEN_COUNT."""
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        text = env.encode("hi")
        expected = [sp1] + text + [sp2]
        actual = [sp1] + text  # missing trailing special
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_COUNT

    def test_special_id_swap(self, env: TokenizerEnv):
        """Same segment structure, but a special token is swapped → SPECIAL_TOKEN_TYPE."""
        sp_a = env.token_id(env.config.known_special_tokens[0])
        sp_b = env.token_id(env.config.known_special_tokens[1])
        text = env.encode("content")
        expected = [sp_a] + text + [sp_a]
        actual = [sp_b] + text + [sp_a]
        result = env.comparator.compare_sequences(expected, actual)
        assert any(m.type == MismatchType.SPECIAL_TOKEN_TYPE and m.segment_index == 0 for m in result)


# ---------------------------------------------------------------------------
# TestContentMismatch — assistant vs non-assistant content classification
#
# When content between two special tokens differs, the comparator checks
# whether it belongs to an assistant turn.  The classification logic is
# model-specific:
#
# Qwen3 template:  <|im_start|>assistant\n{content}<|im_end|>
#   → The comparator decodes the preceding special token + content prefix.
#     If it starts with "assistant_start_str" (= "<|im_start|>assistant"),
#     the segment is classified as assistant content.
#
# GLM47 template:  <|assistant|>{content}<|user|>
#   → Content directly follows the <|assistant|> special token.
#     The decoded "<|assistant|>" matches "assistant_start_str".
#
# Why this matters: ASSISTANT_TEXT mismatches are expected (model generates
# different text than template would) and non-severe.  NON_ASSISTANT_TEXT
# mismatches indicate a bug in the TITO merge — tool responses and system
# messages should be identical.
# ---------------------------------------------------------------------------


class TestContentMismatch:
    def test_non_assistant_content_diff(self, env: TokenizerEnv):
        """Content between non-assistant specials → NON_ASSISTANT_TEXT."""
        # Use index 1 (im_end for Qwen3, user for GLM) — neither is the
        # assistant-start token, so content after it is non-assistant.
        sp = env.token_id(env.config.known_special_tokens[1])
        expected = [sp] + env.encode("Hello world") + [sp]
        actual = [sp] + env.encode("Goodbye world") + [sp]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.NON_ASSISTANT_TEXT

    # -- Qwen3: <|im_start|> + "assistant\n..." vs "user\n..." --

    def test_qwen3_assistant_content_diff(self, qwen3_env: TokenizerEnv):
        """Qwen3: content after <|im_start|>assistant → ASSISTANT_TEXT."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")
        expected = [im_start] + env.encode("assistant\nHello") + [im_end]
        actual = [im_start] + env.encode("assistant\nGoodbye") + [im_end]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.ASSISTANT_TEXT

    def test_qwen3_user_content_diff(self, qwen3_env: TokenizerEnv):
        """Qwen3: content after <|im_start|>user → NON_ASSISTANT_TEXT."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")
        expected = [im_start] + env.encode("user\nHello") + [im_end]
        actual = [im_start] + env.encode("user\nGoodbye") + [im_end]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.NON_ASSISTANT_TEXT

    # -- GLM47: <|assistant|> + content vs <|user|> + content --

    def test_glm_assistant_content_diff(self, glm47_env: TokenizerEnv):
        """GLM: content after <|assistant|> → ASSISTANT_TEXT."""
        env = glm47_env
        asst = env.token_id("<|assistant|>")
        user = env.token_id("<|user|>")
        expected = [asst] + env.encode("Hello") + [user]
        actual = [asst] + env.encode("Goodbye") + [user]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.ASSISTANT_TEXT

    def test_glm_user_content_diff(self, glm47_env: TokenizerEnv):
        """GLM: content after <|user|> → NON_ASSISTANT_TEXT."""
        env = glm47_env
        user = env.token_id("<|user|>")
        asst = env.token_id("<|assistant|>")
        expected = [user] + env.encode("Hello") + [asst]
        actual = [user] + env.encode("Goodbye") + [asst]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.NON_ASSISTANT_TEXT

    # -- Both types in one sequence --

    def test_mixed_assistant_and_non_assistant(self, qwen3_env: TokenizerEnv):
        """A multi-turn sequence can produce both mismatch types simultaneously."""
        env = qwen3_env
        im_start = env.token_id("<|im_start|>")
        im_end = env.token_id("<|im_end|>")
        expected = (
            [im_start] + env.encode("user\nHello") + [im_end] + [im_start] + env.encode("assistant\nHi") + [im_end]
        )
        actual = [im_start] + env.encode("user\nBye") + [im_end] + [im_start] + env.encode("assistant\nBye") + [im_end]
        types = {m.type for m in env.comparator.compare_sequences(expected, actual)}
        assert MismatchType.NON_ASSISTANT_TEXT in types
        assert MismatchType.ASSISTANT_TEXT in types


# ---------------------------------------------------------------------------
# TestTrimTrailing — strip model stop tokens before comparison
#
# Models generate stop tokens (e.g. <|observation|> for GLM, <|im_end|> for
# Qwen) at the end of their output.  The template-rendered "expected" sequence
# may not have these tokens at the same position.  Without trimming, this
# causes a false SPECIAL_TOKEN_COUNT mismatch.
#
# trim_trailing_ids can be set at construction time (stored as default)
# and/or at call time (unioned with the default).  This two-level design
# lets TITOTokenizer set model-specific defaults while callers can add
# situational IDs.
# ---------------------------------------------------------------------------


class TestTrimTrailing:
    def _make(self, env: TokenizerEnv, trim_ids: set[int]) -> TokenSeqComparator:
        return TokenSeqComparator(
            env.tokenizer, assistant_start_str=env.config.assistant_start_str, trim_trailing_ids=trim_ids
        )

    def test_strips_trailing_before_comparison(self, env: TokenizerEnv):
        """Extra trailing stop tokens are stripped so the comparison succeeds."""
        sp = env.token_id(env.config.known_special_tokens[0])
        eos = env.token_id(env.config.known_special_tokens[-1])
        text = env.encode("same content")
        expected = [sp] + text + [sp]
        actual = [sp] + text + [sp, eos, eos]
        # Without trim → mismatch (actual has extra trailing segments)
        assert len(env.comparator.compare_sequences(expected, actual)) > 0
        # With trim → match (trailing eos tokens stripped from both sides)
        assert self._make(env, {eos}).compare_sequences(expected, actual) == []

    def test_does_not_strip_middle(self, env: TokenizerEnv):
        """Trim only affects the tail — same token in the middle is preserved."""
        sp = env.token_id(env.config.known_special_tokens[0])
        eos = env.token_id(env.config.known_special_tokens[-1])
        text = env.encode("content")
        comp = self._make(env, {eos})
        # eos appears in the middle of the sequence — should NOT be stripped
        seq = [sp] + text + [eos] + text + [sp]
        assert comp.compare_sequences(seq, seq) == []

    def test_call_time_union(self, env: TokenizerEnv):
        """trim_trailing_ids at call time is unioned with init-time IDs.

        This lets TITOTokenizer set model defaults at construction while
        callers add situational IDs per comparison.
        """
        sp_tokens = env.config.known_special_tokens
        sp = env.token_id(sp_tokens[0])
        eos1 = env.token_id(sp_tokens[-1])
        eos2 = env.token_id(sp_tokens[1])
        text = env.encode("same")
        expected = [sp] + text + [sp]
        actual = [sp] + text + [sp, eos1, eos2]
        # Init with eos1 only → eos2 still present → mismatch
        comp = self._make(env, {eos1})
        assert len(comp.compare_sequences(expected, actual)) > 0
        # Pass eos2 at call time → union trims both → match
        assert comp.compare_sequences(expected, actual, trim_trailing_ids={eos2}) == []


# ---------------------------------------------------------------------------
# TestGlm47BoundaryTokens — GLM-specific boundary token regression
#
# In GLM47, <|user|> and <|observation|> are both valid next-turn openers
# after an assistant tool call.  The model might stop with <|observation|>
# (expecting a tool response) but the actual next turn starts with <|user|>
# (e.g. when a system message is injected instead).  After TITO's
# trim_trailing_ids strips these from the tail, the *remaining* sequence
# should have the correct boundary token — swapping one for the other at
# a non-tail position is a real structural error.
# ---------------------------------------------------------------------------


class TestGlm47BoundaryTokens:
    def test_user_vs_observation_is_type_mismatch(self, glm47_env: TokenizerEnv):
        """Swapping <|user|> for <|observation|> at the same position is a real error."""
        env = glm47_env
        user_id = env.token_id("<|user|>")
        obs_id = env.token_id("<|observation|>")
        asst_id = env.token_id("<|assistant|>")
        text = env.encode("some content")
        expected = [asst_id] + text + [user_id] + text + [asst_id]
        actual = [asst_id] + text + [obs_id] + text + [asst_id]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_TYPE
        assert result[0].segment_index == 2
