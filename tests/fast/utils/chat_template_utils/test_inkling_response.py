import json

from miles.utils.chat_template_utils.inkling_parser import InklingResponseParser
from miles.utils.chat_template_utils.tito_tokenizer import InklingTITOTokenizer


class FakeInklingTokenizer:
    SPECIAL = {
        "<|endoftext|>": 99,
        "<|message_model|>": 101,
        "<|content_thinking|>": 108,
        "<|content_text|>": 104,
        "<|content_invoke_tool_json|>": 149,
        "<|content_invoke_tool_text|>": 157,
        "<|end_message|>": 110,
        "<|content_model_end_sampling|>": 106,
    }

    all_special_ids = list(SPECIAL.values())
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return self.SPECIAL.get(text, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1000 + ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        inverse = {value: key for key, value in self.SPECIAL.items()}
        return "".join(inverse[token_id] if token_id in inverse else chr(token_id - 1000) for token_id in token_ids)


def _text(tokenizer, value):
    return tokenizer.encode(value, add_special_tokens=False)


def test_parses_ordered_blocks_and_keeps_recipient_separate_from_tool_name():
    tokenizer = FakeInklingTokenizer()
    parser = InklingResponseParser(tokenizer)
    ids = tokenizer.SPECIAL
    payload = json.dumps(
        {"name": "bash_command", "args": {"command": "pwd"}},
        separators=(",", ":"),
    )
    completion = [
        ids["<|content_thinking|>"],
        *_text(tokenizer, "first"),
        ids["<|end_message|>"],
        ids["<|message_model|>"],
        ids["<|content_text|>"],
        *_text(tokenizer, "visible"),
        ids["<|end_message|>"],
        ids["<|message_model|>"],
        *_text(tokenizer, "bash"),
        ids["<|content_invoke_tool_json|>"],
        *_text(tokenizer, payload),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]

    parsed = parser.parse(completion, finish_reason="stop")

    assert parsed.parse_error is None
    assert parsed.client_message["reasoning_content"] == "first"
    assert parsed.client_message["content"] == "visible"
    assert [block["type"] for block in parsed.stored_message["content_blocks"]] == [
        "thinking",
        "text",
        "tool_call",
    ]
    assert parsed.stored_message["content_blocks"][-1]["recipient"] == "bash"
    tool_call = parsed.client_message["tool_calls"][0]
    assert tool_call["function"]["name"] == "bash_command"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert "_miles_raw_completion_token_ids" not in parsed.stored_message


def test_preserves_ordered_blocks_after_flattened_client_replay():
    tokenizer = FakeInklingTokenizer()
    content_blocks = [
        {"type": "thinking", "text": "first"},
        {"type": "text", "text": "visible"},
    ]
    request_messages = [
        {
            "role": "assistant",
            "content": "visible",
            "reasoning_content": "first",
        }
    ]
    stored_messages = [{**request_messages[0], "content_blocks": content_blocks}]

    preserved = InklingTITOTokenizer(tokenizer).preserve_server_message_state(
        stored_messages,
        request_messages,
    )

    assert preserved == [{**request_messages[0], "content_blocks": content_blocks}]
    assert "content_blocks" not in request_messages[0]


def test_marker_spelled_with_ordinary_tokens_remains_payload():
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    completion = [
        ids["<|content_thinking|>"],
        *_text(tokenizer, "literal <|endoftext|> text"),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]

    parsed = InklingResponseParser(tokenizer).parse(completion, finish_reason="stop")

    assert parsed.parse_error is None
    assert parsed.client_message["reasoning_content"] == "literal <|endoftext|> text"


def test_native_special_token_inside_payload_requests_retry():
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    completion = [
        ids["<|content_text|>"],
        ids["<|endoftext|>"],
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]

    parsed = InklingResponseParser(tokenizer).parse(completion, finish_reason="stop")

    assert parsed.parse_error == "control_token_inside_open_block"
    assert "tool_calls" not in parsed.client_message


def test_malformed_and_unterminated_completions_request_retry():
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    malformed = [
        *_text(tokenizer, "bash"),
        ids["<|content_invoke_tool_json|>"],
        *_text(tokenizer, '{"name":"bash_command","args":'),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]
    unterminated = [
        ids["<|content_thinking|>"],
        *_text(tokenizer, "unfinished"),
    ]

    malformed_result = InklingResponseParser(tokenizer).parse(malformed, finish_reason="stop")
    unterminated_result = InklingResponseParser(tokenizer).parse(unterminated, finish_reason="stop")

    assert malformed_result.parse_error == "malformed_tool_json"
    assert malformed_result.frame_error is None
    assert "tool_calls" not in malformed_result.client_message
    assert unterminated_result.parse_error == "unterminated_content_block"
    assert unterminated_result.stored_message["content_blocks"] == []


def _cut_completion(tokenizer, open_part):
    """A complete tool call followed by a message that stops in *open_part*."""
    ids = tokenizer.SPECIAL
    call = [
        *_text(tokenizer, "bash"),
        ids["<|content_invoke_tool_json|>"],
        *_text(tokenizer, '{"name":"bash_command","args":{"command":"pwd"}}'),
        ids["<|end_message|>"],
    ]
    tails = {
        "between": [],
        "header": [ids["<|message_model|>"]],
        "recipient": [ids["<|message_model|>"], *_text(tokenizer, "ba")],
        "thinking": [ids["<|message_model|>"], ids["<|content_thinking|>"], *_text(tokenizer, "so")],
        "text": [ids["<|message_model|>"], ids["<|content_text|>"], *_text(tokenizer, "so")],
        "tool_json": [
            ids["<|message_model|>"],
            *_text(tokenizer, "bash"),
            ids["<|content_invoke_tool_json|>"],
            *_text(tokenizer, '{"name":'),
        ],
    }
    return call + tails[open_part]


_CUT_BLOCKS = {
    "between": None,
    "header": {"type": "text", "text": ""},
    "recipient": {"type": "tool_call", "recipient": "ba", "raw_json": ""},
    "thinking": {"type": "thinking", "text": "so"},
    "text": {"type": "text", "text": "so"},
    "tool_json": {"type": "tool_call", "recipient": "bash", "raw_json": '{"name":'},
}


def test_length_cut_completion_keeps_its_unfinished_message_and_complete_calls():
    tokenizer = FakeInklingTokenizer()
    parser = InklingResponseParser(tokenizer)
    for open_part, cut_block in _CUT_BLOCKS.items():
        parsed = parser.parse(_cut_completion(tokenizer, open_part), finish_reason="length")

        assert parsed.parse_error is None, open_part
        assert parsed.open_part == open_part
        blocks = parsed.stored_message["content_blocks"]
        assert blocks[0]["name"] == "bash_command"
        assert blocks[1:] == ([] if cut_block is None else [cut_block]), open_part
        # The unfinished call is stored as sampled and is not a call.
        assert [call["function"]["name"] for call in parsed.client_message["tool_calls"]] == ["bash_command"]
        assert parsed.client_message.get("reasoning_content") == ("so" if open_part == "thinking" else None)
        assert parsed.client_message["content"] == ("so" if open_part == "text" else "")


def test_length_cut_completion_keeps_its_unfinished_message_after_a_content_error():
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    # A text message with a recipient is a content error; the scan goes on to the cut message.
    completion = [
        *_text(tokenizer, "bash"),
        ids["<|content_text|>"],
        *_text(tokenizer, "hi"),
        ids["<|end_message|>"],
        ids["<|message_model|>"],
        ids["<|content_thinking|>"],
        *_text(tokenizer, "so"),
    ]

    parsed = InklingResponseParser(tokenizer).parse(completion, finish_reason="length")

    assert parsed.parse_error == "unexpected_text_recipient"
    assert parsed.frame_error is None
    assert parsed.stored_message["content_blocks"] == [
        {"type": "text", "text": "hi"},
        {"type": "thinking", "text": "so"},
    ]


def test_closer_finishes_the_open_message_and_ends_the_turn():
    tokenizer = FakeInklingTokenizer()
    tito = InklingTITOTokenizer(tokenizer)
    closers = {
        "between": "<|content_model_end_sampling|>",
        "header": "<|content_text|><|end_message|><|content_model_end_sampling|>",
        "recipient": "<|content_invoke_tool_json|><|end_message|><|content_model_end_sampling|>",
        "thinking": "<|end_message|><|content_model_end_sampling|>",
        "text": "<|end_message|><|content_model_end_sampling|>",
        "tool_json": "<|end_message|><|content_model_end_sampling|>",
    }
    cut_kinds = {"recipient": "tool_call", "tool_json": "tool_call", "thinking": "reasoning"}
    prompt = [tokenizer.SPECIAL["<|message_model|>"]]
    for open_part, closer_text in closers.items():
        cut_turn = prompt + _cut_completion(tokenizer, open_part)
        closer = tito.close_truncated_turn(cut_turn, len(prompt))

        assert tokenizer.decode(closer) == closer_text, open_part
        assert tito.cut_kind(cut_turn, len(prompt)) == cut_kinds.get(open_part, "answer")
        closed = cut_turn + closer
        assert tito.check_turn_structure(closed, (len(prompt), len(closed))) == []
        assert tito.close_truncated_turn(closed, len(prompt)) == []


def test_check_turn_structure_reports_a_broken_frame_before_and_after_closing():
    tokenizer = FakeInklingTokenizer()
    tito = InklingTITOTokenizer(tokenizer)
    ids = tokenizer.SPECIAL
    prompt = [ids["<|message_model|>"]]
    # A control token inside a block stops the scan; the closer cannot repair what the model sampled.
    cut_turn = prompt + [ids["<|content_text|>"], ids["<|endoftext|>"], *_text(tokenizer, "so")]
    closed = cut_turn + tito.close_truncated_turn(cut_turn, len(prompt))

    for turn in (cut_turn, closed):
        assert tito.check_turn_structure(turn, (len(prompt), len(turn))) == ["control_token_inside_open_block"]


def test_parser_instances_have_no_shared_state():
    first_tokenizer = FakeInklingTokenizer()
    second_tokenizer = FakeInklingTokenizer()
    first = InklingResponseParser(first_tokenizer)
    second = InklingResponseParser(second_tokenizer)
    completion = [
        first_tokenizer.SPECIAL["<|content_text|>"],
        *_text(first_tokenizer, "ok"),
        first_tokenizer.SPECIAL["<|end_message|>"],
        first_tokenizer.SPECIAL["<|content_model_end_sampling|>"],
    ]

    assert first.parse(completion, finish_reason="stop").client_message["content"] == "ok"
    assert second.parse(completion, finish_reason="stop").client_message["content"] == "ok"


def test_postprocesses_tool_call_response_and_returns_stored_message():
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    payload = json.dumps(
        {"name": "bash_command", "args": {"command": "pwd"}},
        separators=(",", ":"),
    )
    completion = [
        *_text(tokenizer, "bash"),
        ids["<|content_invoke_tool_json|>"],
        *_text(tokenizer, payload),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]
    assistant_message = {"role": "assistant", "content": "upstream"}
    choice = {
        "message": assistant_message,
        "finish_reason": "stop",
        "meta_info": {"existing": True},
    }

    stored_message = InklingTITOTokenizer(tokenizer).postprocess_completion(
        choice=choice,
        assistant_message=assistant_message,
        completion_token_ids=completion,
    )

    assert choice["finish_reason"] == "tool_calls"
    assert choice["meta_info"] == {
        "existing": True,
        "miles_response_parser": "inkling",
    }
    assert choice["message"]["tool_calls"] == stored_message["tool_calls"]
    assert "_miles_raw_completion_token_ids" not in stored_message
    assert stored_message["content_blocks"][0]["type"] == "tool_call"


def test_postprocesses_parse_error_without_changing_finish_reason():
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    completion = [
        ids["<|content_text|>"],
        ids["<|endoftext|>"],
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]
    assistant_message = {"role": "assistant", "content": "upstream"}
    choice = {
        "message": assistant_message,
        "finish_reason": "stop",
        "meta_info": {"existing": True},
    }

    stored_message = InklingTITOTokenizer(tokenizer).postprocess_completion(
        choice=choice,
        assistant_message=assistant_message,
        completion_token_ids=completion,
    )

    assert choice["finish_reason"] == "stop"
    assert choice["meta_info"] == {
        "existing": True,
        "miles_response_parser": "inkling",
        "miles_response_parse_error": "control_token_inside_open_block",
    }
    assert "tool_calls" not in choice["message"]
    assert "_miles_raw_completion_token_ids" not in stored_message
