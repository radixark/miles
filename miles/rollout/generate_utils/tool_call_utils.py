"""
Utils to handle tool calls.
"""

import json
import uuid
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.utils.types import Sample

_DUMMY_USER = {"role": "user", "content": "dummy"}


def create_tool_call_parser(tool_specs, tool_call_parser):
    return FunctionCallParser(
        tools=TypeAdapter(list[Tool]).validate_python(tool_specs),
        tool_call_parser=tool_call_parser,
    )


async def execute_tool_calls(
    tool_calls: list[ToolCallItem | ChatCompletionMessageToolCall],
    execute_one: Callable,
) -> list[dict[str, Any]]:
    tool_messages = []
    for call in tool_calls:
        tool_messages.append(await _execute_tool_call(call, execute_one))
    return tool_messages


async def _execute_tool_call(
    call: ToolCallItem | ChatCompletionMessageToolCall, execute_one: Callable
) -> dict[str, Any]:
    if isinstance(call, ChatCompletionMessageToolCall):
        name = call.function.name
        params = json.loads(call.function.arguments) if call.function.arguments else {}
        tool_call_id = call.id
    elif isinstance(call, ToolCallItem):
        name = call.name
        params = json.loads(call.parameters) if call.parameters else {}
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
    else:
        raise TypeError(f"Unsupported tool call type: {type(call)}")

    result = await execute_one(name, params)
    assert isinstance(result, str)

    return {"role": "tool", "tool_call_id": tool_call_id, "content": result, "name": name}


def update_sample_with_tool_responses(sample: Sample, tool_messages: list[dict[str, Any]], tokenizer):
    next_obs_tokens_ids: list[int] = tokenize_tool_responses(tool_messages, tokenizer=tokenizer)
    sample.response += tokenizer.decode(next_obs_tokens_ids)
    sample.response_length += len(next_obs_tokens_ids)
    sample.tokens += next_obs_tokens_ids
    sample.loss_mask += [0] * len(next_obs_tokens_ids)
    sample.rollout_log_probs += [0.0] * len(next_obs_tokens_ids)


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
