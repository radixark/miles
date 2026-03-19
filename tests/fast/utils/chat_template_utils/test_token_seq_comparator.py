"""Tests for TokenSeqComparator with real HuggingFace tokenizers.

Test matrix: {Qwen3-4B, GLM-4.7-Flash} × {segmentation, comparison scenarios}.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.token_seq_comparator import MismatchType, Segment, TokenSeqComparator

# ---------------------------------------------------------------------------
# Model configs — one per tokenizer in the test matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    trust_remote_code: bool
    # Text representations — resolved to IDs after loading the tokenizer.
    tool_start_tokens: tuple[str, ...]
    tool_end_tokens: tuple[str, ...]
    # A few known special tokens we assert are recognised.
    known_special_tokens: tuple[str, ...]


_CONFIGS: dict[str, ModelConfig] = {
    "qwen3_4b": ModelConfig(
        model_id="Qwen/Qwen3-4B",
        trust_remote_code=True,
        tool_start_tokens=("<tool_call>", "<tool_response>"),
        tool_end_tokens=("</tool_call>", "</tool_response>"),
        known_special_tokens=("<|im_start|>", "<|im_end|>", "<|endoftext|>"),
    ),
    "glm47_flash": ModelConfig(
        model_id="zai-org/GLM-4.7-Flash",
        trust_remote_code=True,
        tool_start_tokens=("<tool_call>", "<tool_response>"),
        tool_end_tokens=("</tool_call>", "</tool_response>"),
        known_special_tokens=("<|assistant|>", "<|user|>", "<|system|>", "<|observation|>", "<|endoftext|>"),
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class TokenizerEnv:
    """Everything a test needs: tokenizer + pre-built comparator + token IDs."""

    tokenizer: AutoTokenizer
    config: ModelConfig
    comparator: TokenSeqComparator
    tool_start_ids: set[int]
    tool_end_ids: set[int]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_id(self, token_text: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token_text)


_ENV_CACHE: dict[str, TokenizerEnv] = {}


def _build_env(cfg: ModelConfig) -> TokenizerEnv:
    if cfg.model_id in _ENV_CACHE:
        return _ENV_CACHE[cfg.model_id]

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    tool_start_ids = {tok.convert_tokens_to_ids(t) for t in cfg.tool_start_tokens}
    tool_end_ids = {tok.convert_tokens_to_ids(t) for t in cfg.tool_end_tokens}

    # Tool tokens may not be marked special=True in the vocab — include them
    # in special_token_ids so that segmentation treats them as boundaries.
    extra_special = tool_start_ids | tool_end_ids
    comp = TokenSeqComparator(
        tok,
        special_token_ids=TokenSeqComparator._collect_special_ids(tok) | extra_special,
        tool_start_ids=tool_start_ids,
        tool_end_ids=tool_end_ids,
    )

    env = TokenizerEnv(
        tokenizer=tok,
        config=cfg,
        comparator=comp,
        tool_start_ids=tool_start_ids,
        tool_end_ids=tool_end_ids,
    )
    _ENV_CACHE[cfg.model_id] = env
    return env


@pytest.fixture(params=list(_CONFIGS.keys()))
def env(request) -> TokenizerEnv:
    return _build_env(_CONFIGS[request.param])


# ===========================================================================
# _collect_special_ids
# ===========================================================================


class TestCollectSpecialIds:
    def test_known_specials_are_collected(self, env: TokenizerEnv):
        """All tokens the model documents as special are in _collect_special_ids."""
        collected = TokenSeqComparator._collect_special_ids(env.tokenizer)
        for tok_text in env.config.known_special_tokens:
            tid = env.token_id(tok_text)
            assert tid in collected, f"{tok_text} (id={tid}) not in collected special ids"

    def test_regular_tokens_excluded(self, env: TokenizerEnv):
        """Ordinary text tokens are NOT in the special set."""
        collected = TokenSeqComparator._collect_special_ids(env.tokenizer)
        for tid in env.encode("Hello world"):
            assert tid not in collected, f"regular token id={tid} should not be special"


# ===========================================================================
# segment_by_special_tokens
# ===========================================================================


class TestSegmentation:
    def test_empty(self, env: TokenizerEnv):
        assert env.comparator.segment_by_special_tokens([]) == []

    def test_plain_text_single_segment(self, env: TokenizerEnv):
        ids = env.encode("The quick brown fox jumps over the lazy dog.")
        segs = env.comparator.segment_by_special_tokens(ids)
        assert len(segs) == 1
        assert segs[0].is_special is False
        assert segs[0].token_ids == ids

    def test_special_tokens_create_boundaries(self, env: TokenizerEnv):
        """<special> text <special> → 3 segments (special, content, special)."""
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        text_ids = env.encode("some content")
        seq = [sp1] + text_ids + [sp2]
        segs = env.comparator.segment_by_special_tokens(seq)
        assert len(segs) == 3
        assert segs[0] == Segment(token_ids=[sp1], is_special=True)
        assert segs[1] == Segment(token_ids=text_ids, is_special=False)
        assert segs[2] == Segment(token_ids=[sp2], is_special=True)

    def test_consecutive_specials(self, env: TokenizerEnv):
        """Multiple adjacent specials each get their own segment."""
        sp_ids = [env.token_id(t) for t in env.config.known_special_tokens[:3]]
        segs = env.comparator.segment_by_special_tokens(sp_ids)
        assert len(segs) == len(sp_ids)
        assert all(s.is_special for s in segs)
        for seg, expected_id in zip(segs, sp_ids, strict=False):
            assert seg.token_ids == [expected_id]

    def test_tool_call_structure(self, env: TokenizerEnv):
        """<tool_call> json_content </tool_call> → 3 segments."""
        tc_start = env.token_id(env.config.tool_start_tokens[0])
        tc_end = env.token_id(env.config.tool_end_tokens[0])
        json_ids = env.encode('{"name":"get_weather","arguments":{"city":"Shanghai"}}')
        seq = [tc_start] + json_ids + [tc_end]
        segs = env.comparator.segment_by_special_tokens(seq)
        assert len(segs) == 3
        assert segs[0] == Segment(token_ids=[tc_start], is_special=True)
        assert segs[1] == Segment(token_ids=json_ids, is_special=False)
        assert segs[2] == Segment(token_ids=[tc_end], is_special=True)

    def test_realistic_multi_segment(self, env: TokenizerEnv):
        """A realistic assistant turn: <sp> role <tool_call> json </tool_call> <sp>."""
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        tc_start = env.token_id(env.config.tool_start_tokens[0])
        tc_end = env.token_id(env.config.tool_end_tokens[0])
        role_ids = env.encode("assistant\n")
        json_ids = env.encode('{"name":"f","arguments":{}}')
        seq = [sp1] + role_ids + [tc_start] + json_ids + [tc_end] + [sp2]
        segs = env.comparator.segment_by_special_tokens(seq)
        # sp1, role_content, tc_start, json_content, tc_end, sp2
        assert len(segs) == 6
        assert segs[0].is_special and segs[0].token_ids == [sp1]
        assert not segs[1].is_special and segs[1].token_ids == role_ids
        assert segs[2].is_special and segs[2].token_ids == [tc_start]
        assert not segs[3].is_special and segs[3].token_ids == json_ids
        assert segs[4].is_special and segs[4].token_ids == [tc_end]
        assert segs[5].is_special and segs[5].token_ids == [sp2]


# ===========================================================================
# compare_sequences — identical
# ===========================================================================


class TestCompareIdentical:
    def test_identical_plain_text(self, env: TokenizerEnv):
        sp = env.token_id(env.config.known_special_tokens[0])
        ids = [sp] + env.encode("Hello world") + [sp]
        assert env.comparator.compare_sequences(ids, ids) == []

    def test_identical_with_tool_call(self, env: TokenizerEnv):
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        json_ids = env.encode('{"name":"get_weather","arguments":{"city":"Shanghai"}}')
        seq = [tc_s] + json_ids + [tc_e]
        assert env.comparator.compare_sequences(seq, seq) == []

    def test_both_empty(self, env: TokenizerEnv):
        assert env.comparator.compare_sequences([], []) == []


# ===========================================================================
# compare_sequences — SPECIAL_TOKEN mismatches
# ===========================================================================


class TestCompareSpecialToken:
    def test_different_segment_count(self, env: TokenizerEnv):
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        text_ids = env.encode("hi")
        expected = [sp1] + text_ids + [sp2]
        actual = [sp1] + text_ids  # missing trailing special
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_COUNT
        assert result[0].segment_index == -1
        assert "segment count differs" in result[0].detail

    def test_structure_pattern_differs(self, env: TokenizerEnv):
        sp = env.token_id(env.config.known_special_tokens[0])
        text_ids = env.encode("hello")
        # expected: [special, content, special]
        # actual:   [content, special, content] (same length but different pattern)
        expected = [sp] + text_ids + [sp]
        actual = text_ids + [sp] + text_ids
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) >= 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_COUNT

    def test_special_id_differs(self, env: TokenizerEnv):
        """Same structure, but one special token is swapped for another."""
        sp_tokens = env.config.known_special_tokens
        if len(sp_tokens) < 2:
            pytest.skip("Need at least 2 known special tokens")
        sp_a = env.token_id(sp_tokens[0])
        sp_b = env.token_id(sp_tokens[1])
        text_ids = env.encode("content")
        expected = [sp_a] + text_ids + [sp_a]
        actual = [sp_b] + text_ids + [sp_a]
        result = env.comparator.compare_sequences(expected, actual)
        assert any(m.type == MismatchType.SPECIAL_TOKEN_TYPE and m.segment_index == 0 for m in result)


# ===========================================================================
# compare_sequences — TEXT mismatches
# ===========================================================================


class TestCompareText:
    def test_content_text_differs(self, env: TokenizerEnv):
        sp = env.token_id(env.config.known_special_tokens[0])
        expected = [sp] + env.encode("Hello world") + [sp]
        actual = [sp] + env.encode("Goodbye world") + [sp]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.TEXT
        assert result[0].segment_index == 1

    def test_content_length_differs(self, env: TokenizerEnv):
        sp = env.token_id(env.config.known_special_tokens[0])
        expected = [sp] + env.encode("short") + [sp]
        actual = [sp] + env.encode("a much longer sentence here") + [sp]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.TEXT

    def test_non_tool_content_between_tool_tokens_without_config(self, env: TokenizerEnv):
        """Comparator without tool config → always TEXT even between tool tokens."""
        comp_no_tool = TokenSeqComparator(
            env.tokenizer,
            special_token_ids=env.comparator._special_ids,
        )
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        expected = [tc_s] + env.encode("foo") + [tc_e]
        actual = [tc_s] + env.encode("bar") + [tc_e]
        result = comp_no_tool.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.TEXT


# ===========================================================================
# compare_sequences — JSON (tool) mismatches
# ===========================================================================


_JSON_COMPACT = '{"name":"get_weather","arguments":{"city":"Shanghai"}}'
_JSON_PRETTY = '{\n  "name": "get_weather",\n  "arguments": {"city": "Shanghai"}\n}'
_JSON_DIFFERENT = '{"name":"get_weather","arguments":{"city":"Beijing"}}'
_NOT_JSON = "This is not JSON at all"


class TestCompareJson:
    def test_json_formatting_only(self, env: TokenizerEnv):
        """Compact vs pretty-printed JSON → JSON mismatch (formatting)."""
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        expected = [tc_s] + env.encode(_JSON_COMPACT) + [tc_e]
        actual = [tc_s] + env.encode(_JSON_PRETTY) + [tc_e]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.JSON
        assert "formatting" in result[0].detail

    def test_json_semantically_different(self, env: TokenizerEnv):
        """Shanghai vs Beijing → JSON mismatch (parsed JSON differs)."""
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        expected = [tc_s] + env.encode(_JSON_COMPACT) + [tc_e]
        actual = [tc_s] + env.encode(_JSON_DIFFERENT) + [tc_e]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.JSON
        assert "parsed JSON differs" in result[0].detail

    def test_json_parse_failure_falls_back_to_text(self, env: TokenizerEnv):
        """One side isn't valid JSON → falls back to TEXT mismatch."""
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        expected = [tc_s] + env.encode(_JSON_COMPACT) + [tc_e]
        actual = [tc_s] + env.encode(_NOT_JSON) + [tc_e]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.TEXT
        assert "JSON parsing failed" in result[0].detail

    def test_tool_response_json(self, env: TokenizerEnv):
        """Tool response tokens also trigger JSON comparison."""
        tr_s = env.token_id(env.config.tool_start_tokens[1])  # <tool_response>
        tr_e = env.token_id(env.config.tool_end_tokens[1])  # </tool_response>
        compact = '{"result":"sunny","temp":25}'
        pretty = '{\n  "result": "sunny",\n  "temp": 25\n}'
        expected = [tr_s] + env.encode(compact) + [tr_e]
        actual = [tr_s] + env.encode(pretty) + [tr_e]
        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.JSON

    def test_identical_tool_content_no_mismatch(self, env: TokenizerEnv):
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        json_ids = env.encode(_JSON_COMPACT)
        seq = [tc_s] + json_ids + [tc_e]
        assert env.comparator.compare_sequences(seq, seq) == []


# ===========================================================================
# trim_trailing_ids
# ===========================================================================


class TestTrimTrailingIds:
    def test_trim_removes_trailing_specials(self, env: TokenizerEnv):
        """Trailing eos-like tokens are stripped before comparison."""
        sp = env.token_id(env.config.known_special_tokens[0])
        eos = env.token_id(env.config.known_special_tokens[-1])
        text_ids = env.encode("same content")
        expected = [sp] + text_ids + [sp]
        actual = [sp] + text_ids + [sp, eos, eos]  # extra trailing eos
        # Without trim → segment count differs
        result_no_trim = env.comparator.compare_sequences(expected, actual)
        assert len(result_no_trim) > 0
        # With trim → matches
        result_trim = env.comparator.compare_sequences(expected, actual, trim_trailing_ids={eos})
        assert result_trim == []

    def test_trim_does_not_affect_middle(self, env: TokenizerEnv):
        """trim_trailing_ids only strips from the end, not the middle."""
        sp = env.token_id(env.config.known_special_tokens[0])
        eos = env.token_id(env.config.known_special_tokens[-1])
        text_ids = env.encode("content")
        # eos in the middle should not be stripped
        seq = [sp] + text_ids + [eos] + text_ids + [sp]
        result = env.comparator.compare_sequences(seq, seq, trim_trailing_ids={eos})
        assert result == []


# ===========================================================================
# _is_tool_segment edge cases with real tokens
# ===========================================================================


class TestIsToolSegment:
    def test_content_preceded_by_tool_start(self, env: TokenizerEnv):
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        json_ids = env.encode('{"a":1}')
        segs = env.comparator.segment_by_special_tokens([tc_s] + json_ids + [tc_e])
        # seg 0 = <tool_call>, seg 1 = json content, seg 2 = </tool_call>
        assert env.comparator._is_tool_segment(segs, 1)

    def test_content_after_non_tool_special(self, env: TokenizerEnv):
        """Content after a non-tool special token → not a tool segment."""
        sp = env.token_id(env.config.known_special_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        text_ids = env.encode("just text")
        segs = env.comparator.segment_by_special_tokens([sp] + text_ids + [tc_e])
        assert not env.comparator._is_tool_segment(segs, 1)

    def test_tool_start_no_matching_end(self, env: TokenizerEnv):
        """<tool_call> content <non_tool_end> → not a tool segment (end token mismatch)."""
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        sp = env.token_id(env.config.known_special_tokens[0])  # not a tool-end token
        text_ids = env.encode("data")
        segs = env.comparator.segment_by_special_tokens([tc_s] + text_ids + [sp])
        assert not env.comparator._is_tool_segment(segs, 1)

    def test_no_tool_config(self, env: TokenizerEnv):
        """Comparator without tool_start_ids → _is_tool_segment always False."""
        comp_no_tool = TokenSeqComparator(
            env.tokenizer,
            special_token_ids=env.comparator._special_ids,
        )
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        json_ids = env.encode('{"a":1}')
        segs = comp_no_tool.segment_by_special_tokens([tc_s] + json_ids + [tc_e])
        assert not comp_no_tool._is_tool_segment(segs, 1)


# ===========================================================================
# Mixed mismatches — realistic sequences
# ===========================================================================


class TestCompareMixed:
    def test_text_and_json_mismatch_in_one_sequence(self, env: TokenizerEnv):
        """A sequence with both a plain text diff and a tool call JSON diff."""
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])

        expected = [sp1] + env.encode("Hello") + [tc_s] + env.encode(_JSON_COMPACT) + [tc_e] + [sp2]
        actual = [sp1] + env.encode("Goodbye") + [tc_s] + env.encode(_JSON_DIFFERENT) + [tc_e] + [sp2]

        result = env.comparator.compare_sequences(expected, actual)
        types = {m.type for m in result}
        assert MismatchType.TEXT in types
        assert MismatchType.JSON in types

    def test_full_realistic_sequence_no_diff(self, env: TokenizerEnv):
        """Full assistant turn with tool call — identical → no mismatches."""
        sp1 = env.token_id(env.config.known_special_tokens[0])
        sp2 = env.token_id(env.config.known_special_tokens[1])
        tc_s = env.token_id(env.config.tool_start_tokens[0])
        tc_e = env.token_id(env.config.tool_end_tokens[0])
        tr_s = env.token_id(env.config.tool_start_tokens[1])
        tr_e = env.token_id(env.config.tool_end_tokens[1])

        seq = (
            [sp1]
            + env.encode("assistant\n")
            + [tc_s]
            + env.encode(_JSON_COMPACT)
            + [tc_e]
            + [sp2]
            + [sp1]
            + env.encode("tool\n")
            + [tr_s]
            + env.encode('{"result":"ok"}')
            + [tr_e]
            + [sp2]
        )
        assert env.comparator.compare_sequences(seq, seq) == []


# ===========================================================================
# GLM 4.7 specific: <|user|> vs <|observation|> type mismatch is acceptable
# ===========================================================================


@pytest.fixture
def glm47_env() -> TokenizerEnv:
    return _build_env(_CONFIGS["glm47_flash"])


class TestGlm47SpecialTokenType:
    """After stripping stop tokens from pretokenized (in sessions.py), all
    special-token type differences are fatal for every model — no exceptions.
    """

    def test_user_vs_observation_is_type_mismatch(self, glm47_env: TokenizerEnv):
        """Swapping <|user|> for <|observation|> at the same position → TYPE mismatch."""
        env = glm47_env
        user_id = env.token_id("<|user|>")
        obs_id = env.token_id("<|observation|>")
        assistant_id = env.token_id("<|assistant|>")
        text_ids = env.encode("some content")

        expected = [assistant_id] + text_ids + [user_id] + text_ids + [assistant_id]
        actual = [assistant_id] + text_ids + [obs_id] + text_ids + [assistant_id]

        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_TYPE
        assert result[0].segment_index == 2

    def test_count_mismatch_on_missing_segment(self, glm47_env: TokenizerEnv):
        """Extra or missing segments → SPECIAL_TOKEN_COUNT."""
        env = glm47_env
        user_id = env.token_id("<|user|>")
        assistant_id = env.token_id("<|assistant|>")
        text_ids = env.encode("content")

        expected = [assistant_id] + text_ids + [user_id]
        actual = [assistant_id] + text_ids  # missing <|user|>

        result = env.comparator.compare_sequences(expected, actual)
        assert len(result) == 1
        assert result[0].type == MismatchType.SPECIAL_TOKEN_COUNT
