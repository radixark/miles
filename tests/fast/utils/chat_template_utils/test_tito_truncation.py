"""L1 checks for closing turns cut by ``finish_reason='length'``.

Every family with a ``TRUNCATION_GRAMMAR`` renders each fixture assistant turn
with its own template, tokenizes it as the sampled output, and cuts it at every
token.  At each cut:

- the closer emits exactly the closing markers the template still renders for
  the structures open at the cut, then the end-of-turn token.  The expectation
  pairs the markers of the complete render, independently of the closer's scan
  of the truncated prefix;
- the closed turn passes ``check_turn_structure``;
- appending a user or tool turn through ``merge_tokens`` keeps every sampled
  token, and the result matches the template render in special tokens and
  non-assistant text.

Separate checks pin each grammar's closing text to its template and require
every added token the template puts inside an assistant turn to be a declared
marker or explicitly ignored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["rollout"])

from miles.utils.chat_template_utils import resolve_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import (
    TITOTokenizer,
    TITOTokenizerType,
    extract_template_args,
    get_tito_tokenizer,
)
from miles.utils.chat_template_utils.tito_tokenizer.truncation import MarkerPair, TruncationGrammar
from miles.utils.chat_template_utils.token_seq_comparator import MismatchType
from miles.utils.processing_utils import load_tokenizer

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

_DEEPSEEK_TOKENIZER = "deepseek-ai/DeepSeek-V3.2"
_DEEPSEEK_MODEL_TYPES = {
    TITOTokenizerType.DEEPSEEKV32: "deepseek_v32",
    TITOTokenizerType.DEEPSEEKV4: "deepseek_v4",
    TITOTokenizerType.DEEPSEEKV41: "deepseek_v41",
}


@dataclass(frozen=True)
class _Family:
    tito_type: TITOTokenizerType
    model_id: str
    # chat_template_kwargs that turn reasoning off, when the template allows it.
    thinking_off: dict[str, Any] | None = None
    # Added tokens the template puts inside an assistant turn that close nothing.
    ignored_added_tokens: frozenset[str] = field(default_factory=frozenset)


_ENABLE_THINKING_OFF = {"enable_thinking": False}
_THINKING_OFF = {"thinking": False}
_KIMI_SEPARATOR = frozenset({"<|tool_call_argument_begin|>"})

_FAMILIES: dict[str, _Family] = {
    "qwen3": _Family(TITOTokenizerType.QWEN3, "Qwen/Qwen3-4B", _ENABLE_THINKING_OFF),
    "qwennext": _Family(TITOTokenizerType.QWENNEXT, "Qwen/Qwen3-4B-Thinking-2507"),
    "qwen35": _Family(TITOTokenizerType.QWEN35, "Qwen/Qwen3.5-0.8B", _ENABLE_THINKING_OFF),
    "qwen36": _Family(TITOTokenizerType.QWEN36, "Qwen/Qwen3.6-35B-A3B", _ENABLE_THINKING_OFF),
    "qwen38small": _Family(TITOTokenizerType.QWEN38_SMALL, "Qwen/Qwen3.8-27B", _ENABLE_THINKING_OFF),
    "nemotron3": _Family(
        TITOTokenizerType.NEMOTRON3, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", _ENABLE_THINKING_OFF
    ),
    "glm47": _Family(TITOTokenizerType.GLM47, "zai-org/GLM-4.7-Flash", _ENABLE_THINKING_OFF),
    "glm53": _Family(TITOTokenizerType.GLM53, "zai-org/GLM-5.3"),
    "kimi25": _Family(TITOTokenizerType.KIMI25, "moonshotai/Kimi-K2.5", _THINKING_OFF, _KIMI_SEPARATOR),
    "kimi26": _Family(TITOTokenizerType.KIMI26, "moonshotai/Kimi-K2.6", _THINKING_OFF, _KIMI_SEPARATOR),
    "minimax_m25": _Family(TITOTokenizerType.MINIMAX_M25, "MiniMaxAI/MiniMax-M2.5"),
    "minimax_m27": _Family(TITOTokenizerType.MINIMAX_M27, "MiniMaxAI/MiniMax-M2.7"),
    # V4 and V4.1 reuse V3.2's tokenizer; the markers under test are the same text in all three encoders.
    "deepseekv32": _Family(TITOTokenizerType.DEEPSEEKV32, _DEEPSEEK_TOKENIZER, _THINKING_OFF),
    "deepseekv4": _Family(TITOTokenizerType.DEEPSEEKV4, _DEEPSEEK_TOKENIZER, _THINKING_OFF),
    "deepseekv41": _Family(TITOTokenizerType.DEEPSEEKV41, _DEEPSEEK_TOKENIZER, _THINKING_OFF),
}


def _load_tito(family: _Family, tmp_path_factory) -> TITOTokenizer:
    if family.tito_type in _DEEPSEEK_MODEL_TYPES:
        # DeepSeek rendering is routed by the checkpoint's config.json model_type.
        checkpoint = tmp_path_factory.mktemp(family.tito_type.value)
        load_tokenizer(family.model_id).save_pretrained(checkpoint)
        (checkpoint / "config.json").write_text(json.dumps({"model_type": _DEEPSEEK_MODEL_TYPES[family.tito_type]}))
        tokenizer = load_tokenizer(str(checkpoint))
    else:
        tokenizer = load_tokenizer(
            family.model_id,
            chat_template_path=resolve_fixed_chat_template(family.tito_type)[0],
            trust_remote_code=True,
        )
    return get_tito_tokenizer(tokenizer, family.tito_type)


@pytest.fixture(scope="module", params=list(_FAMILIES))
def family_tito(request, tmp_path_factory) -> tuple[_Family, TITOTokenizer]:
    family = _FAMILIES[request.param]
    return family, _load_tito(family, tmp_path_factory)


# ---------------------------------------------------------------------------
# Fixture conversations
# ---------------------------------------------------------------------------


def _tool(name: str, parameter: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Look up {parameter}.",
            "parameters": {"type": "object", "properties": {parameter: {"type": "string"}}, "required": [parameter]},
        },
    }


_TOOLS = [_tool("get_weather", "city"), _tool("get_date", "timezone")]
_SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
_USER = {"role": "user", "content": "What is the weather in Paris, and the date in UTC?"}
_NEXT_USER = {"role": "user", "content": "Thanks. And tomorrow?"}
_REASONING = "The user asks for the weather and the date, so I need both tools."
_CALLS = [
    {
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
    },
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_date", "arguments": json.dumps({"timezone": "UTC"})},
    },
]
_TOOL_RESULTS = [
    {"role": "tool", "tool_call_id": "call_0", "content": "Sunny, 21 degrees."},
    {"role": "tool", "tool_call_id": "call_1", "content": "2026-10-10"},
]


def _assistant(content: str, *, reasoning: str | None = _REASONING, tool_calls=None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


_TWO_CALLS = _assistant("Let me check both.", tool_calls=_CALLS)

# name -> (reasoning on, conversation before the turn, the turn as stored)
_FIXTURES: dict[str, tuple[bool, list[dict[str, Any]], dict[str, Any]]] = {
    "answer": (True, [_SYSTEM, _USER], _assistant("It is sunny in Paris.")),
    "reasoning_only": (True, [_SYSTEM, _USER], _assistant("")),
    "one_call": (True, [_SYSTEM, _USER], _assistant("Checking.", tool_calls=_CALLS[:1])),
    "two_calls": (True, [_SYSTEM, _USER], _TWO_CALLS),
    "calls_only": (True, [_SYSTEM, _USER], _assistant("", tool_calls=_CALLS)),
    # The sampled text adds prose after the first tool-call block (see _build_case).
    "text_after_call": (True, [_SYSTEM, _USER], _TWO_CALLS),
    # The reasoning quotes a tool-call block (see _build_case).
    "call_in_reasoning": (True, [_SYSTEM, _USER], _assistant("Done.")),
    "after_tool_turn": (True, [_SYSTEM, _USER, _TWO_CALLS, *_TOOL_RESULTS], _assistant("Sunny, on 2026-10-10.")),
    # An earlier message quotes an opening marker; it must not open the new turn's reasoning.
    "user_quotes_marker": (
        True,
        [_SYSTEM, {"role": "user", "content": "Why do your replies start with <think>?"}],
        _assistant("It marks my reasoning."),
    ),
    "answer_thinking_off": (False, [_SYSTEM, _USER], _assistant("It is sunny in Paris.", reasoning=None)),
    "two_calls_thinking_off": (
        False,
        [_SYSTEM, _USER],
        _assistant("Let me check.", reasoning=None, tool_calls=_CALLS),
    ),
}


@dataclass(frozen=True)
class _Case:
    template_args: dict[str, Any]
    messages: list[dict[str, Any]]
    assistant: dict[str, Any]
    # Prompt text that opens the turn, after the rendered conversation.
    generation_prompt: str
    prompt_ids: list[int]
    # The turn's sampled text, through its end-of-turn token.
    completion: str


def _encode(tito: TITOTokenizer, text: str) -> list[int]:
    return tito.tokenizer.encode(text, add_special_tokens=False)


def _render(tito: TITOTokenizer, messages, *, add_generation_prompt: bool, template_args) -> str:
    return tito.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, template_args=template_args)


def _outer_block_text(grammar: TruncationGrammar, completion: str) -> tuple[int, int]:
    """Span of the first outermost tool-call block in *completion*."""
    outer = grammar.tool_blocks[0]
    start = completion.index(outer.open)
    return start, completion.index(outer.close_marker, start) + len(outer.close_marker)


def _build_case(tito: TITOTokenizer, family: _Family, name: str) -> _Case:
    reasoning_on, messages, assistant = _FIXTURES[name]
    if not reasoning_on and family.thinking_off is None:
        pytest.skip(f"{family.tito_type.value} cannot turn reasoning off")
    grammar = tito.TRUNCATION_GRAMMAR
    if name == "call_in_reasoning":
        calls = _build_case(tito, family, "two_calls").completion
        start, end = _outer_block_text(grammar, calls)
        assistant = {**assistant, "reasoning_content": f"Draft: {calls[start:end]} Then answer."}

    request = {"chat_template_kwargs": {} if reasoning_on else dict(family.thinking_off), "tools": _TOOLS}
    args = extract_template_args(tito.resolve_request_args(request, turn_args=None))
    history = _render(tito, messages, add_generation_prompt=False, template_args=args)
    prompt = _render(tito, messages, add_generation_prompt=True, template_args=args)
    with_turn = _render(tito, messages + [assistant], add_generation_prompt=False, template_args=args)
    assert prompt.startswith(history) and with_turn.startswith(prompt), "template does not extend its prompt"
    completion = with_turn[len(prompt) :].rstrip()

    if name == "text_after_call":
        _, end = _outer_block_text(grammar, completion)
        completion = completion[:end] + " Note: two calls." + completion[end:]
    return _Case(
        template_args=args,
        messages=messages,
        assistant=assistant,
        generation_prompt=prompt[len(history) :],
        prompt_ids=_encode(tito, prompt),
        completion=completion,
    )


# ---------------------------------------------------------------------------
# Expected closers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Span:
    pair: MarkerPair
    open_end: int
    close_start: int
    close_end: int


def _marker_pattern(markers) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(marker) for marker in sorted(set(markers), key=len, reverse=True)))


def _marker_spans(grammar: TruncationGrammar, generation_prompt: str, completion: str) -> list[_Span]:
    """Pair every structure marker of a complete, well-formed turn render.

    A generation prompt that turns reasoning off may render a lone closing
    marker (GLM-4.7, DeepSeek chat mode); it closes nothing.
    """
    roles = {}
    for pair in (grammar.think, *grammar.tool_blocks):
        roles[pair.open] = (pair, True)
        roles[pair.close_marker] = (pair, False)
    text = generation_prompt + completion
    spans, opened = [], []
    for match in _marker_pattern(roles).finditer(text):
        pair, is_open = roles[match.group()]
        if is_open:
            opened.append((pair, match.end()))
            continue
        if not opened and match.end() <= len(generation_prompt):
            continue
        open_pair, open_end = opened.pop()
        assert open_pair == pair, f"render is not well formed at offset {match.start()}: {text!r}"
        spans.append(_Span(pair, open_end, match.start(), match.end()))
    assert not opened, f"render leaves structures open: {text!r}"
    return spans


def _expected_closing_markers(grammar: TruncationGrammar, spans: list[_Span], cut: int) -> list[str]:
    """Closing markers the template renders after *cut* for structures opened before it."""
    open_spans = sorted((s for s in spans if s.open_end <= cut < s.close_end), key=lambda s: s.close_start)
    return [s.pair.close_marker for s in open_spans] + ([grammar.end_of_turn] if grammar.end_of_turn else [])


def _closing_markers(grammar: TruncationGrammar, closer_text: str) -> list[str]:
    """The markers in *closer_text*, which must hold nothing else but whitespace."""
    markers = [pair.close_marker for pair in (grammar.think, *grammar.tool_blocks)]
    pattern = _marker_pattern(markers + ([grammar.end_of_turn] if grammar.end_of_turn else []))
    assert pattern.sub("", closer_text).strip() == "", f"closer adds more than markers: {closer_text!r}"
    return pattern.findall(closer_text)


def _split_to_characters(tito: TITOTokenizer, token_ids: list[int]) -> list[int]:
    """Re-tokenize every non-added token one character at a time, as an unusual sampled split."""
    added = tito.tokenizer.added_tokens_decoder
    split: list[int] = []
    for token_id in token_ids:
        if token_id in added:
            split.append(token_id)
        else:
            for char in tito.tokenizer.decode([token_id]):
                split.extend(_encode(tito, char))
    return split


# ---------------------------------------------------------------------------
# Cut at every token
# ---------------------------------------------------------------------------

_HARD_MISMATCHES = {MismatchType.SPECIAL_TOKEN_COUNT, MismatchType.SPECIAL_TOKEN_TYPE, MismatchType.NON_ASSISTANT_TEXT}


def _close_every_cut(tito: TITOTokenizer, case: _Case, sampled: list[int]) -> list[list[int]]:
    """Close the turn at every cut and check each closer; return the closed sequences."""
    grammar = tito.TRUNCATION_GRAMMAR
    spans = _marker_spans(grammar, case.generation_prompt, case.completion)
    start = len(case.prompt_ids)
    # A complete turn ends with its end-of-turn token, which a truncated turn never samples.
    last_cut = len(sampled) if grammar.end_of_turn is None else len(sampled) - 1
    closed_turns = []
    for cut in range(last_cut + 1):
        sampled_text = tito.tokenizer.decode(sampled[:cut])
        assert case.completion.startswith(sampled_text)
        token_ids = case.prompt_ids + sampled[:cut]
        closer = tito.close_truncated_turn(token_ids, start)

        expected = _expected_closing_markers(grammar, spans, len(case.generation_prompt) + len(sampled_text))
        assert _closing_markers(grammar, tito.tokenizer.decode(closer)) == expected, f"cut after {sampled_text!r}"
        closed = token_ids + closer
        assert tito.check_turn_structure(closed, (start, len(closed))) == [], f"cut after {sampled_text!r}"
        closed_turns.append(closed)
    return closed_turns


def _check_appended_turn(tito: TITOTokenizer, case: _Case, closed_turns: list[list[int]], appended) -> None:
    old_messages = case.messages + [case.assistant]
    new_messages = old_messages + appended
    expected = _encode(tito, _render(tito, new_messages, add_generation_prompt=True, template_args=case.template_args))
    comparator = tito.create_comparator()
    # The appended messages are the same at every cut, so render their tokens once.
    incremental = tito.tokenize_additional_messages(old_messages, new_messages, template_args=case.template_args)
    with patch.object(tito, "tokenize_additional_messages", return_value=incremental):
        for closed in closed_turns:
            merged = tito.merge_tokens(old_messages, new_messages, closed, template_args=case.template_args)
            assert merged[: len(closed)] == closed
            hard = [m for m in comparator.compare_sequences(expected, merged) if m.type in _HARD_MISMATCHES]
            assert not hard, [m.to_dict() for m in hard]


@pytest.mark.parametrize("fixture", list(_FIXTURES))
def test_close_at_every_cut(family_tito, fixture):
    family, tito = family_tito
    case = _build_case(tito, family, fixture)
    sampled = _encode(tito, case.completion)

    closed_turns = _close_every_cut(tito, case, sampled)
    _check_appended_turn(tito, case, closed_turns, [_NEXT_USER])
    if case.assistant.get("tool_calls"):
        _check_appended_turn(tito, case, closed_turns, _TOOL_RESULTS[: len(case.assistant["tool_calls"])])

    _close_every_cut(tito, case, _split_to_characters(tito, sampled))


# ---------------------------------------------------------------------------
# Grammar against the template
# ---------------------------------------------------------------------------


def _leading_whitespace(text: str) -> str:
    return text[: len(text) - len(text.lstrip())]


def test_closing_text_matches_template(family_tito):
    family, tito = family_tito
    grammar = tito.TRUNCATION_GRAMMAR
    end_of_turn = grammar.end_of_turn or ""

    # Reasoning with nothing after it ends with the declared closing text.
    assert _build_case(tito, family, "reasoning_only").completion.endswith(
        _REASONING + grammar.think.close + end_of_turn
    )

    # Each closing marker follows exactly the declared whitespace; the outermost
    # tool-call block ends the turn with the declared trailing whitespace.
    completion = _build_case(tito, family, "two_calls").completion
    for pair in (grammar.think, *grammar.tool_blocks):
        lead = _leading_whitespace(pair.close)
        positions = [m.start() for m in re.finditer(re.escape(pair.close_marker), completion)]
        assert positions, f"{pair.close_marker!r} not rendered"
        for position in positions:
            before = completion[:position]
            assert before.endswith(lead) and not before[: len(before) - len(lead)][-1:].isspace(), pair
    outer = grammar.tool_blocks[0]
    assert completion.endswith(outer.close_marker + outer.close[len(outer.close.rstrip()) :] + end_of_turn)


@pytest.mark.parametrize("fixture", ["two_calls", "call_in_reasoning"])
def test_added_tokens_in_turn_are_declared(family_tito, fixture):
    family, tito = family_tito
    grammar = tito.TRUNCATION_GRAMMAR
    markers = [marker for pair in (grammar.think, *grammar.tool_blocks) for marker in (pair.open, pair.close_marker)]
    markers += [grammar.end_of_turn] if grammar.end_of_turn else []
    added = tito.tokenizer.added_tokens_decoder
    for token_id in _encode(tito, _build_case(tito, family, fixture).completion):
        if token_id in added:
            content = added[token_id].content
            assert content in family.ignored_added_tokens or any(content in marker for marker in markers), content


# ---------------------------------------------------------------------------
# Registration and the structure check
# ---------------------------------------------------------------------------


def test_every_family_but_default_and_inkling_has_a_grammar():
    unsupported = {TITOTokenizerType.DEFAULT, TITOTokenizerType.INKLING}
    for tito_type in TITOTokenizerType:
        cls = TITOTokenizerType.get_tokenizer_class(tito_type)
        assert (cls.TRUNCATION_GRAMMAR is None) == (tito_type in unsupported), tito_type
    with pytest.raises(ValueError, match="cannot close a truncated turn"):
        TITOTokenizer(MagicMock()).close_truncated_turn([1, 2], 1)


@pytest.fixture(scope="module")
def qwen3_tito(tmp_path_factory) -> TITOTokenizer:
    return _load_tito(_FAMILIES["qwen3"], tmp_path_factory)


@pytest.mark.parametrize(
    "completion,problem",
    [
        ("<think>\nR\n</think>\n\nC", "turn does not end with '<|im_end|>'"),
        ("<think>\nR<|im_end|>", "unclosed: '</think>'"),
        ("<think>\nR\n</think>\n\n<tool_call>\n{}<|im_end|>", "unclosed: '</tool_call>'"),
        ("<think>\n<tool_call>\n</think>\n\nC<|im_end|>", "'</think>' at offset 20 does not close the innermost"),
        ("C\n</tool_call><|im_end|>", "'</tool_call>' at offset 2 does not close the innermost"),
    ],
)
def test_check_turn_structure_reports_problems(qwen3_tito, completion, problem):
    prompt_ids = _encode(qwen3_tito, "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n")
    token_ids = prompt_ids + _encode(qwen3_tito, completion)
    problems = qwen3_tito.check_turn_structure(token_ids, (len(prompt_ids), len(token_ids)))
    assert any(p.startswith(problem) for p in problems), problems


# ---------------------------------------------------------------------------
# Grammar scan semantics
# ---------------------------------------------------------------------------

_TOOL_CALL = MarkerPair("<tool_call>", "\n</tool_call>")
_GRAMMAR = TruncationGrammar(
    think=MarkerPair("<think>", "\n</think>\n\n"), tool_blocks=(_TOOL_CALL,), end_of_turn="<e>"
)


def test_reasoning_opened_by_generation_prompt():
    assert _GRAMMAR.opened_by("<|im_start|>assistant\n<think>\n") == (_GRAMMAR.think,)
    assert _GRAMMAR.opened_by("<|im_start|>assistant\n<think>\n\n</think>\n\n") == ()
    assert _GRAMMAR.opened_by("<|im_start|>assistant\n") == ()


def test_closing_marker_closes_structures_opened_inside_it():
    stack, problems = _GRAMMAR.scan("a <tool_call> b </think> c <tool_call> d", (_GRAMMAR.think,))
    assert stack == [_TOOL_CALL] and len(problems) == 1


def test_closing_marker_without_open_structure_is_ignored():
    stack, problems = _GRAMMAR.scan("a </tool_call> b", ())
    assert stack == [] and len(problems) == 1


def test_closing_text_skips_whitespace_already_sampled():
    assert _GRAMMAR.closing_text("<think>\nR\n", [_GRAMMAR.think]) == "</think>\n\n<e>"
    assert _GRAMMAR.closing_text("<think>\nR", [_GRAMMAR.think, _TOOL_CALL]) == "\n</tool_call>\n</think>\n\n<e>"
