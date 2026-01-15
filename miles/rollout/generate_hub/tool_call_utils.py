from typing import Any


def tokenize_tool_response(
    message: dict[str, Any],
    tokenizer,
) -> list[int]:
    """
    Tokenize a tool response message by applying chat template diff.

    This function computes the token IDs for a tool response by:
    1. Creating messages with dummy user, dummy assistant, and the tool response
    2. Applying chat template and tokenizing
    3. Removing the tool response and tokenizing again
    4. Computing the diff to get only the tool response tokens

    Args:
        message: A tool response message dict with keys like:
            - "role": "tool"
            - "content": the tool execution result
            - "tool_call_id": the ID matching the assistant's tool call
            - "name": (optional) the function name
        tokenizer: A tokenizer with apply_chat_template method

    Returns:
        List of token IDs corresponding to the tool response
    """
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
