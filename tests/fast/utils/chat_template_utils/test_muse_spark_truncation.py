"""L1 checks for closing Muse Spark turns cut by ``finish_reason='length'``.

A Muse Spark turn is a run of messages, and ``<|start|>``, ``<|eom|>`` and
``<|eot|>`` are comparator boundaries, so closing the tokens is not enough: the
stored message must hold the cut turn too.  Each fixture turn is rendered by the
fixed template, tokenized as the sampled output, and cut at every token.  At
each cut:

- the closer is the one the cut's place in the complete turn calls for, read
  from the complete turn's text independently of the closer;
- ``cut_kind`` names the kind of message the cut is in, and the closed turn
  passes ``check_turn_structure``;
- the message ``postprocess_completion`` stores for the cut completion renders
  the cut turn followed by its closer;
- appending the next message through ``merge_tokens`` keeps every token of the
  closed turn, and the result matches the template render.

The stored view holds text, so where the last sampled token and the closer
re-tokenize differently at their junction, the render is text-equal but not
token-equal (``_TEXT_ONLY_CUTS``, in a message header or an ATEM block).  There the comparator must still find no special-token or
non-assistant difference.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.utils.chat_template_utils.tito_families import load_family_tito

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, TITOTokenizerType, extract_template_args
from miles.utils.chat_template_utils.token_seq_comparator import MismatchType

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


def _assistant(content: str, *, reasoning: str | None = _REASONING, tool_calls=None, **extra) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, **extra}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


_TWO_CALLS = _assistant("Let me check both.", tool_calls=_CALLS)
# ATEM text in reasoning is not a tool call.
_ATEM_TEXT = (
    '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>'
)

# name -> (conversation before the turn, the turn)
_FIXTURES: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
    "reasoning_answer": ([_SYSTEM, _USER], _assistant("It is sunny in Paris.")),
    "reasoning_only": ([_SYSTEM, _USER], _assistant("")),
    "reasoning_commentary_call": ([_SYSTEM, _USER], _assistant("Checking.", tool_calls=_CALLS[:1])),
    "two_calls": ([_SYSTEM, _USER], _TWO_CALLS),
    "calls_only": ([_SYSTEM, _USER], _assistant("", reasoning=None, tool_calls=_CALLS)),
    "answer_only": ([_SYSTEM, _USER], _assistant("It is sunny in Paris.", reasoning=None)),
    "commentary_only": ([_SYSTEM, _USER], _assistant("Working on it.", reasoning=None, recipient="commentary")),
    "call_in_reasoning": ([_SYSTEM, _USER], _assistant("Done.", reasoning=f"Draft: {_ATEM_TEXT} Then answer.")),
    "after_tool_turn": ([_SYSTEM, _USER, _TWO_CALLS, *_TOOL_RESULTS], _assistant("Sunny, on 2026-10-10.")),
}
# Cuts whose stored view re-renders text-equal but not token-equal: the last sampled
# token and the closer merge into other tokens when the closed text is tokenized whole.
_TEXT_ONLY_CUTS = {
    "reasoning_answer": 0,
    "reasoning_only": 0,
    "reasoning_commentary_call": 8,
    "two_calls": 15,
    "calls_only": 14,
    "answer_only": 0,
    "commentary_only": 1,
    "call_in_reasoning": 0,
    "after_tool_turn": 0,
}
_HARD = {MismatchType.SPECIAL_TOKEN_COUNT, MismatchType.SPECIAL_TOKEN_TYPE, MismatchType.NON_ASSISTANT_TEXT}

_HEADER = "assistant to="
_FINAL_MESSAGE = "<|start|>assistant to=user<|message|><|eot|>"
_MESSAGE_RE = re.compile(r"(?:<\|start\|>)?(.*?)<\|message\|>(.*?)(<\|eom\|>|<\|eot\|>)", re.S)
_ATEM_RE = re.compile(r'<atem:(\w+)(?: name="[^"]*")?>|</atem:(\w+)>')
# element -> (closing tag, text after it, whether it starts on a new line), as render_atem writes them
_ATEM_CLOSE = {
    "parameter": ("</atem:parameter>", "\n", False),
    "invoke": ("</atem:invoke>", "\n", True),
    "function_calls": ("</atem:function_calls>", "", True),
}


@pytest.fixture(scope="module")
def tito() -> TITOTokenizer:
    return load_family_tito(TITOTokenizerType.MUSE_SPARK)


def _encode(tito: TITOTokenizer, text: str) -> list[int]:
    return tito.tokenizer.encode(text, add_special_tokens=False)


def _decode(tito: TITOTokenizer, ids: list[int]) -> str:
    return tito.tokenizer.decode(ids, skip_special_tokens=False)


def _render_ids(tito: TITOTokenizer, messages, template_args, *, add_generation_prompt: bool = False) -> list[int]:
    return _encode(
        tito,
        tito.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, template_args=template_args),
    )


def _expected_atem_closer(body: str, cut: int) -> str:
    """The closer of the complete, well-formed ATEM *body* cut after *cut* characters.

    Open elements are those whose opening tag ends by *cut* and whose closing
    tag ends after it; a closing tag cut after at least ``</`` is completed.
    """
    spans: list[tuple[str, int, int, int]] = []  # (element, open end, close start, close end)
    stack: list[tuple[str, int]] = []
    for match in _ATEM_RE.finditer(body):
        if match.group(1):
            stack.append((match.group(1), match.end()))
        else:
            element, open_end = stack.pop()
            spans.append((element, open_end, match.start(), match.end()))
    open_elements = sorted((s for s in spans if s[1] <= cut < s[3]), key=lambda s: s[1])
    closer = ""
    if open_elements and open_elements[-1][2] + 2 <= cut:
        element, _, close_start, _ = open_elements.pop()
        tag, after, _ = _ATEM_CLOSE[element]
        closer = tag[cut - close_start :] + after
    for element, *_ in reversed(open_elements):
        tag, after, on_new_line = _ATEM_CLOSE[element]
        closer += ("\n" if on_new_line and not (body[:cut] + closer).endswith("\n") else "") + tag + after
    return closer


def _expected(completion: str, cut: int) -> tuple[str, str]:
    """The closer text and cut kind for the complete turn's text *completion* cut after *cut* characters."""
    for index, match in enumerate(_MESSAGE_RE.finditer(completion)):
        header = ("assistant" if index == 0 else "") + completion[match.start(1) : min(cut, match.end(1))]
        recipient = match.group(1).removeprefix("assistant").removeprefix(" to=")
        if match.start(1) <= cut <= match.end(1):
            if _HEADER.startswith(header):
                return _HEADER[len(header) :] + "user<|message|><|eot|>", "answer"
            partial = header[len(_HEADER) :]
            kind = "answer" if any(r.startswith(partial) for r in ("self", "user", "commentary")) else "tool_call"
            return "<|message|>" + ("<|eom|>" if partial in ("self", "commentary") else "<|eot|>"), kind
        if match.start(2) <= cut <= match.end(2):
            if recipient == "self":
                return "<|eom|>", "reasoning"
            if recipient in ("user", "commentary"):
                # The template ends a commentary message with <|eom|>, also as the turn's last.
                return ("<|eom|>" if recipient == "commentary" else "<|eot|>"), "answer"
            return _expected_atem_closer(match.group(2), cut - match.start(2)) + "<|eot|>", "tool_call"
        if cut == match.end():
            return ("" if recipient in ("self", "commentary") else _FINAL_MESSAGE), "answer"
    raise AssertionError(f"cut {cut} is not in a message of {completion!r}")


def _store(tito: TITOTokenizer, sampled: list[int], *, finish_reason: str) -> dict[str, Any]:
    """The message the session stores for *sampled*.

    The client message is a plausible one built by hand; the stored view does
    not depend on it.
    """
    choice = {"finish_reason": finish_reason, "message": {"role": "assistant", "content": _decode(tito, sampled)}}
    return tito.postprocess_completion(
        choice=choice, assistant_message=choice["message"], completion_token_ids=sampled
    )


def _hard(mismatches) -> list:
    return [m for m in mismatches if m.type in _HARD]


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
    completion = _decode(tito, sampled)
    start = len(prompt_ids)
    comparator = tito.create_comparator()

    # A turn that ends with <|eot|> is never cut after it; one that ends with <|eom|> can be.
    last_cut = len(sampled) - 1 if completion.endswith("<|eot|>") else len(sampled)
    text_only = 0
    for cut in range(last_cut + 1):
        cut_turn = prompt_ids + sampled[:cut]
        cut_text = _decode(tito, sampled[:cut])
        label = f"cut after {cut_text!r}"
        assert completion.startswith(cut_text), label
        closer_text, cut_kind = _expected(completion, len(cut_text))

        closer = tito.close_truncated_turn(cut_turn, start)
        assert _decode(tito, closer) == closer_text, label
        assert tito.cut_kind(cut_turn, start) == cut_kind, label
        closed = cut_turn + closer
        assert tito.check_turn_structure(closed, (start, len(closed))) == [], label
        assert tito.close_truncated_turn(closed, start) == [], label

        stored = _store(tito, sampled[:cut], finish_reason="length")
        rendered = _render_ids(tito, messages + [stored], args)
        assert _decode(tito, rendered) == _decode(tito, closed), label
        assert _hard(comparator.compare_sequences(rendered, closed)) == [], label
        text_only += rendered != closed

        new_messages = messages + [stored, _NEXT_USER]
        merged = tito.merge_tokens(messages + [stored], new_messages, closed, template_args=args)
        assert merged[: len(closed)] == closed, label
        expected = _render_ids(tito, new_messages, args, add_generation_prompt=True)
        mismatches = comparator.compare_sequences(expected, merged)
        assert (mismatches == []) if rendered == closed else (_hard(mismatches) == []), label
    assert text_only == _TEXT_ONLY_CUTS[fixture]


def test_complete_turn_stores_the_parsed_message(tito):
    messages, assistant = _FIXTURES["two_calls"]
    args = extract_template_args(
        tito.resolve_request_args({"chat_template_kwargs": {}, "tools": _TOOLS}, turn_args=None)
    )
    prompt_ids = _render_ids(tito, messages, args, add_generation_prompt=True)
    sampled = _render_ids(tito, messages + [assistant], args)[len(prompt_ids) :]
    choice = {"finish_reason": "stop", "message": assistant}
    stored = tito.postprocess_completion(choice=choice, assistant_message=assistant, completion_token_ids=sampled)
    assert stored is assistant


def test_check_turn_structure_reports_what_a_cut_leaves_open(tito):
    messages, assistant = _FIXTURES["two_calls"]
    args = extract_template_args(
        tito.resolve_request_args({"chat_template_kwargs": {}, "tools": _TOOLS}, turn_args=None)
    )
    prompt_ids = _render_ids(tito, messages, args, add_generation_prompt=True)
    sampled = _render_ids(tito, messages + [assistant], args)[len(prompt_ids) :]
    start = len(prompt_ids)
    eom = tito.tokenizer.convert_tokens_to_ids("<|eom|>")

    def problems(turn: list[int]) -> list[str]:
        return tito.check_turn_structure(prompt_ids + turn, (start, start + len(turn)))

    assert problems(sampled) == []
    assert problems(sampled[:-1]) == ["unclosed: to=get_date message"]
    assert problems(sampled[:-1] + [eom]) == ["turn does not end with <|eot|>"]
    assert problems(sampled[:1]) == ["unclosed: message header"]
    # A frame the model broke stays broken after closing, and the closer adds nothing.
    broken = sampled[:3] + [tito.tokenizer.convert_tokens_to_ids("<|start|>")]
    assert problems(broken) == ["<|start|> in a message body"]
    assert tito.close_truncated_turn(prompt_ids + broken, start) == []


def test_closer_counts_an_opening_tag_with_more_than_its_name(tito):
    prompt = _encode(tito, "<|start|>assistant")
    cut = _encode(tito, ' to=get_weather<|message|><atem:function_calls>\n<atem:invoke  name="get_weather" id="1">\n')
    cut += _encode(tito, '<atem:parameter name="city">Par')

    closer = tito.close_truncated_turn(prompt + cut, len(prompt))

    assert _decode(tito, closer) == "</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>"


def test_text_that_spells_a_control_token_stays_text_in_the_stored_turn(tito):
    messages, _ = _FIXTURES["answer_only"]
    args = extract_template_args(
        tito.resolve_request_args({"chat_template_kwargs": {}, "tools": _TOOLS}, turn_args=None)
    )
    prompt_ids = _render_ids(tito, messages, args, add_generation_prompt=True)
    # The model spells the control token with ordinary tokens.
    spelled = [token_id for piece in ("<", "|eot|", ">") for token_id in _encode(tito, piece)]
    sampled = _encode(tito, " to=user<|message|>Use ") + spelled
    closed = prompt_ids + sampled + tito.close_truncated_turn(prompt_ids + sampled, len(prompt_ids))

    stored = _store(tito, sampled, finish_reason="length")
    rendered = _render_ids(tito, messages + [stored], args)

    assert _hard(tito.create_comparator().compare_sequences(rendered, closed)) == []
