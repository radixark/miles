from typing import Any


_DUMMY_USER = {"role": "user", "content": "dummy"}


def tokenize_tool_responses(
    tool_messages: list[dict[str, Any]],
    tokenizer,
) -> list[int]:
    dummy_assistant = _build_dummy_assistant(tool_messages)
    base_messages = [_DUMMY_USER, dummy_assistant]

    messages_without = base_messages
    messages_with = base_messages + tool_messages

    tokens_with = tokenizer.apply_chat_template(messages_with, tokenize=True, add_generation_prompt=False)
    tokens_without = tokenizer.apply_chat_template(messages_without, tokenize=True, add_generation_prompt=False)

    assert tokens_with.startswith(tokens_without), f"{tokens_with=} {tokens_without=}"
    return tokens_with[len(tokens_without) :]


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": resp.get("tool_call_id", f"call_dummy_{i}"),
                "type": "function",
                "function": {
                    "name": resp.get("name", "dummy_func"),
                    "arguments": "{}",
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }
