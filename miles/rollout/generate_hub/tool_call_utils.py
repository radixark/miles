from typing import Any


_DUMMY_USER = {"role": "user", "content": "dummy"}


# TODO: very naive implementation, need the to-be-implemented e2e test to validate.
def tokenize_tool_responses(
    tool_messages: list[dict[str, Any]],
    tokenizer,
) -> list[int]:
    return _tokenize_postfix_messages(tool_messages, tokenizer)


def _tokenize_postfix_messages(
    postfix_messages: list[dict[str, Any]],
    tokenizer,
) -> list[int]:
    dummy_assistant = _build_dummy_assistant(postfix_messages)
    base_messages = [_DUMMY_USER, dummy_assistant]

    messages_without = base_messages
    messages_with = base_messages + postfix_messages

    tokens_with = tokenizer.apply_chat_template(messages_with, tokenize=True, add_generation_prompt=True)
    tokens_without = tokenizer.apply_chat_template(messages_without, tokenize=True, add_generation_prompt=False)

    assert tokens_with[: len(tokens_without)] == tokens_without, (
        f"Fail to tokenize_tool_responses caused by token prefix mismatch. "
        f"This can happen for thinking model or models with special chat template, "
        f"and this simple example does not support it yet, "
        f"since this means we cannot have a append-only token id list. "
        f"{tokens_with=} {tokens_without=} "
        f"{tokenizer.decode(tokens_with)=} {tokenizer.decode(tokens_without)=} "
    )
    return tokens_with[len(tokens_without) :]


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": resp.get("tool_call_id", f"call0000{i}"),
                "type": "function",
                "function": {
                    "name": resp.get("name", "dummy_func"),
                    "arguments": {},
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }
