from typing import Any


def tokenize_tool_response(
    message: dict[str, Any],
    tokenizer,
) -> list[int]:
    dummy_user = {"role": "user", "content": "dummy"}
    dummy_assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": message.get("tool_call_id", "call_dummy"),
                "type": "function",
                "function": {
                    "name": message.get("name", "dummy_func"),
                    "arguments": "{}",
                },
            }
        ],
    }

    messages_with_tool = [dummy_user, dummy_assistant, message]
    messages_without_tool = [dummy_user, dummy_assistant]

    tokens_with_tool = tokenizer.apply_chat_template(
        messages_with_tool, tokenize=True, add_generation_prompt=False
    )
    tokens_without_tool = tokenizer.apply_chat_template(
        messages_without_tool, tokenize=True, add_generation_prompt=False
    )

    assert tokens_with_tool[: len(tokens_without_tool)] == tokens_without_tool, (
        "Token prefix mismatch: the tokens without tool should be a prefix of tokens with tool"
    )

    return tokens_with_tool[len(tokens_without_tool) :]
