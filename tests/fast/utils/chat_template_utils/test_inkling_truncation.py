"""L1 checks for closing Inkling turns cut by ``finish_reason='length'``.

An Inkling turn is a run of messages, and every message start is a comparator
boundary, so closing the tokens is not enough: the stored message must hold the
unfinished message too.  Each fixture turn is rendered by the fixed template,
tokenized as the sampled output, and cut at every token.  At each cut:

- the closer is the one the last control token before the cut calls for, read
  from the sampled tokens independently of the parser;
- the message ``postprocess_completion`` stores for the cut completion renders,
  token for token, the cut turn followed by its closer;
- ``cut_kind`` names the kind of block the cut is in, and the closed turn
  passes ``check_turn_structure``;
- appending the next message through ``merge_tokens`` keeps every token of the
  closed turn, and the result matches the template render.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.utils.chat_template_utils.tito_families import load_family_tito

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, TITOTokenizerType, extract_template_args

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["rollout"])


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
# The fixed template renders a call's arguments from a parsed object.
_CALLS = [
    {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
    {"id": "call_1", "type": "function", "function": {"name": "get_date", "arguments": {"timezone": "UTC"}}},
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

# name -> (conversation before the turn, the turn)
_FIXTURES: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
    "answer": ([_SYSTEM, _USER], _assistant("It is sunny in Paris.")),
    "reasoning_only": ([_SYSTEM, _USER], _assistant("")),
    "answer_without_reasoning": ([_SYSTEM, _USER], _assistant("It is sunny in Paris.", reasoning=None)),
    "two_calls": ([_SYSTEM, _USER], _TWO_CALLS),
    "calls_only": ([_SYSTEM, _USER], _assistant("", reasoning=None, tool_calls=_CALLS)),
    "after_tool_turn": ([_SYSTEM, _USER, _TWO_CALLS, *_TOOL_RESULTS], _assistant("Sunny, on 2026-10-10.")),
}

_HEADER = "<|message_model|>"
_END_MESSAGE = "<|end_message|>"
_END_TURN = "<|content_model_end_sampling|>"
# content token -> the cut kind of its block
_CONTENT_TOKENS = {
    "<|content_thinking|>": "reasoning",
    "<|content_text|>": "answer",
    "<|content_invoke_tool_json|>": "tool_call",
}


@pytest.fixture(scope="module")
def tito() -> TITOTokenizer:
    return load_family_tito(TITOTokenizerType.INKLING)


def _encode(tito: TITOTokenizer, text: str) -> list[int]:
    return tito.tokenizer.encode(text, add_special_tokens=False)


def _render_ids(tito: TITOTokenizer, messages, template_args, *, add_generation_prompt: bool = False) -> list[int]:
    return _encode(
        tito,
        tito.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, template_args=template_args),
    )


def _expected(tito: TITOTokenizer, sampled: list[int]) -> tuple[str, str]:
    """The closer text and the cut kind after *sampled*, from its last control token.

    The generation prompt ends with a message start, so a cut before any
    control token is in the first message's header.
    """
    control = _HEADER
    after_control = 0
    for token_id in sampled:
        text = tito.tokenizer.decode([token_id])
        if text in (_HEADER, _END_MESSAGE, *_CONTENT_TOKENS):
            control, after_control = text, 0
        else:
            after_control += 1
    if control == _END_MESSAGE:
        return _END_TURN, "answer"
    if control in _CONTENT_TOKENS:
        return _END_MESSAGE + _END_TURN, _CONTENT_TOKENS[control]
    # A header that holds text is a tool recipient; the closer ends it as an empty call.
    if after_control:
        return "<|content_invoke_tool_json|>" + _END_MESSAGE + _END_TURN, "tool_call"
    return "<|content_text|>" + _END_MESSAGE + _END_TURN, "answer"


def _store(tito: TITOTokenizer, sampled: list[int], *, finish_reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The message the session stores for *sampled*, and the choice the client gets."""
    choice = {"finish_reason": finish_reason, "message": {"role": "assistant", "content": ""}}
    stored = tito.postprocess_completion(
        choice=choice, assistant_message=choice["message"], completion_token_ids=sampled
    )
    return stored, choice


@pytest.mark.parametrize("fixture", list(_FIXTURES))
def test_close_at_every_cut(tito, fixture):
    messages, assistant = _FIXTURES[fixture]
    args = extract_template_args(
        tito.resolve_request_args({"chat_template_kwargs": {}, "tools": _TOOLS}, turn_args=None)
    )
    prompt_ids = _render_ids(tito, messages, args, add_generation_prompt=True)
    with_turn = _render_ids(tito, messages + [assistant], args)
    assert with_turn[: len(prompt_ids)] == prompt_ids, "template does not extend its prompt"
    sampled = with_turn[len(prompt_ids) :]
    assert tito.tokenizer.decode(sampled[-1:]) == _END_TURN
    start = len(prompt_ids)
    comparator = tito.create_comparator()

    # A complete turn ends with the end-of-turn token, which a truncated turn never samples.
    for cut in range(len(sampled)):
        cut_turn = prompt_ids + sampled[:cut]
        label = f"cut after {tito.tokenizer.decode(sampled[:cut])!r}"
        closer_text, cut_kind = _expected(tito, sampled[:cut])

        closer = tito.close_truncated_turn(cut_turn, start)
        assert tito.tokenizer.decode(closer) == closer_text, label
        assert tito.cut_kind(cut_turn, start) == cut_kind, label
        closed = cut_turn + closer
        assert tito.check_turn_structure(closed, (start, len(closed))) == [], label

        stored, choice = _store(tito, sampled[:cut], finish_reason="length")
        assert "miles_response_parse_error" not in choice["meta_info"], label
        assert _render_ids(tito, messages + [stored], args) == closed, label

        new_messages = messages + [stored, _NEXT_USER]
        merged = tito.merge_tokens(messages + [stored], new_messages, closed, template_args=args)
        assert merged[: len(closed)] == closed, label
        expected = _render_ids(tito, new_messages, args, add_generation_prompt=True)
        assert comparator.compare_sequences(expected, merged) == [], label


def test_cut_turn_returns_its_complete_calls_only(tito):
    messages, assistant = _FIXTURES["two_calls"]
    args = extract_template_args(
        tito.resolve_request_args({"chat_template_kwargs": {}, "tools": _TOOLS}, turn_args=None)
    )
    prompt_ids = _render_ids(tito, messages, args, add_generation_prompt=True)
    sampled = _render_ids(tito, messages + [assistant], args)[len(prompt_ids) :]
    # Drop the end of the turn, the second call's end and the last token of its JSON.
    cut = sampled[:-3]
    raw_json = tito.tokenizer.decode(cut).rpartition("<|content_invoke_tool_json|>")[2]
    assert '"get_date"' in raw_json and not raw_json.endswith("}}")
    stored, choice = _store(tito, cut, finish_reason="length")

    assert choice["finish_reason"] == "length"
    assert [call["function"]["name"] for call in choice["message"]["tool_calls"]] == ["get_weather"]
    assert stored["content_blocks"][-1] == {
        "type": "tool_call",
        "recipient": "get_date",
        "raw_json": raw_json,
    }

    # The same tokens from a turn that was not cut are malformed, as before.
    _, choice = _store(tito, cut, finish_reason="stop")
    assert choice["meta_info"]["miles_response_parse_error"] == "unterminated_content_block"
    assert "tool_calls" not in choice["message"]


def test_check_turn_structure_reports_what_a_cut_leaves_open(tito):
    messages, assistant = _FIXTURES["two_calls"]
    args = extract_template_args(
        tito.resolve_request_args({"chat_template_kwargs": {}, "tools": _TOOLS}, turn_args=None)
    )
    prompt_ids = _render_ids(tito, messages, args, add_generation_prompt=True)
    sampled = _render_ids(tito, messages + [assistant], args)[len(prompt_ids) :]
    start = len(prompt_ids)

    def problems(cut: int) -> list[str]:
        return tito.check_turn_structure(prompt_ids + sampled[:cut], (start, start + cut))

    assert problems(len(sampled)) == []
    assert problems(len(sampled) - 1) == [f"turn does not end with {_END_TURN}"]
    assert problems(len(sampled) - 3) == ["unclosed: tool_json"]
    assert problems(2) == ["unclosed: thinking"]
    assert problems(0) == ["unclosed: header"]
