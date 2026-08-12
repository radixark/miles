"""Anthropic Messages API translation for the Miles session server.

The session server records OpenAI chat-completion requests because that is the
format consumed by the trajectory and sample builders.  Agents such as Claude
Code speak Anthropic's Messages API instead.  This module keeps the wire-format
translation independent from HTTP and session state so it can be tested in
isolation.
"""

import json
from collections.abc import Mapping
from typing import Any


class AnthropicProtocolError(ValueError):
    """Raised when an Anthropic request cannot be represented faithfully."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnthropicProtocolError(f"{name} must be an object")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnthropicProtocolError(f"{name} must be a non-empty string")
    return value


def _image_part(source: Any) -> dict[str, Any]:
    source = _require_mapping(source, "image.source")
    source_type = source.get("type")
    if source_type == "base64":
        data = _require_nonempty_string(source.get("data"), "image.source.data")
        media_type = source.get("media_type", "image/png")
        if not isinstance(media_type, str) or not media_type:
            raise AnthropicProtocolError("image.source.media_type must be a non-empty string")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        url = _require_nonempty_string(source.get("url"), "image.source.url")
    else:
        raise AnthropicProtocolError(f"unsupported image source type: {source_type!r}")
    return {"type": "image_url", "image_url": {"url": url}}


def _content_value(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and parts[0].get("type") == "text":
        return str(parts[0].get("text", ""))
    return parts


def _tool_result_content(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise AnthropicProtocolError("tool_result.content must be a string or an array")

    parts: list[dict[str, Any]] = []
    for item_index, raw_item in enumerate(content):
        item = _require_mapping(raw_item, f"tool_result.content[{item_index}]")
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise AnthropicProtocolError(f"tool_result.content[{item_index}].text must be a string")
            parts.append({"type": "text", "text": text})
        elif item_type == "image":
            parts.append(_image_part(item.get("source")))
        else:
            raise AnthropicProtocolError(f"unsupported tool_result content block type: {item_type!r}")
    return _content_value(parts) if parts else ""


def _append_content_message(messages: list[dict[str, Any]], role: str, parts: list[dict[str, Any]]) -> None:
    if not parts:
        return
    messages.append({"role": role, "content": _content_value(parts.copy())})
    parts.clear()


def _convert_message_content(
    messages: list[dict[str, Any]],
    role: str,
    content: Any,
    message_index: int,
) -> None:
    if isinstance(content, str):
        messages.append({"role": role, "content": content})
        return
    if not isinstance(content, list):
        raise AnthropicProtocolError(f"messages[{message_index}].content must be a string or an array")

    content_parts: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    emitted_tool_result = False

    for block_index, raw_block in enumerate(content):
        block = _require_mapping(raw_block, f"messages[{message_index}].content[{block_index}]")
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise AnthropicProtocolError(f"messages[{message_index}].content[{block_index}].text must be a string")
            content_parts.append({"type": "text", "text": text})
        elif block_type == "image":
            content_parts.append(_image_part(block.get("source")))
        elif block_type == "thinking":
            if role != "assistant":
                raise AnthropicProtocolError("thinking blocks are only valid in assistant messages")
            thinking = block.get("thinking")
            if not isinstance(thinking, str):
                raise AnthropicProtocolError(
                    f"messages[{message_index}].content[{block_index}].thinking must be a string"
                )
            reasoning_parts.append(thinking)
        elif block_type == "redacted_thinking":
            raise AnthropicProtocolError("redacted_thinking history is not supported")
        elif block_type == "tool_use":
            if role != "assistant":
                raise AnthropicProtocolError("tool_use blocks are only valid in assistant messages")
            tool_id = _require_nonempty_string(block.get("id"), "tool_use.id")
            tool_name = _require_nonempty_string(block.get("name"), "tool_use.name")
            tool_input = block.get("input", {})
            if not isinstance(tool_input, Mapping):
                raise AnthropicProtocolError("tool_use.input must be an object")
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            )
        elif block_type == "tool_result":
            if role != "user":
                raise AnthropicProtocolError("tool_result blocks are only valid in user messages")
            _append_content_message(messages, role, content_parts)
            tool_use_id = _require_nonempty_string(block.get("tool_use_id"), "tool_result.tool_use_id")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": _tool_result_content(block.get("content")),
                }
            )
            emitted_tool_result = True
        else:
            raise AnthropicProtocolError(f"unsupported Anthropic content block type: {block_type!r}")

    if role == "assistant":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": _content_value(content_parts) if content_parts else "",
        }
        if reasoning_parts:
            message["reasoning_content"] = "\n".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)
    else:
        _append_content_message(messages, role, content_parts)
        if not content and not emitted_tool_result:
            messages.append({"role": role, "content": ""})


def _convert_system(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        raise AnthropicProtocolError("system must be a string or an array")

    text_parts: list[str] = []
    for block_index, raw_block in enumerate(system):
        block = _require_mapping(raw_block, f"system[{block_index}]")
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise AnthropicProtocolError("system content blocks must be text blocks")
        text_parts.append(block["text"])
    return "\n".join(text_parts)


def _convert_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise AnthropicProtocolError("tools must be an array")

    converted: list[dict[str, Any]] = []
    for tool_index, raw_tool in enumerate(tools):
        tool = _require_mapping(raw_tool, f"tools[{tool_index}]")
        name = _require_nonempty_string(tool.get("name"), f"tools[{tool_index}].name")
        schema = tool.get("input_schema")
        if not isinstance(schema, Mapping):
            raise AnthropicProtocolError(
                f"tools[{tool_index}].input_schema must be an object; Anthropic server-side tools are not supported"
            )
        converted_tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description") or "",
                "parameters": dict(schema),
            },
        }
        if "defer_loading" in tool:
            converted_tool["defer_loading"] = bool(tool["defer_loading"])
        converted.append(converted_tool)
    return converted


def _convert_tool_choice(tool_choice: Any, tools: list[dict[str, Any]] | None) -> Any:
    if tool_choice is None:
        return "auto" if tools else None
    tool_choice = _require_mapping(tool_choice, "tool_choice")
    choice_type = tool_choice.get("type")
    if choice_type in ("auto", "none"):
        return choice_type
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        name = _require_nonempty_string(tool_choice.get("name"), "tool_choice.name")
        return {"type": "function", "function": {"name": name}}
    raise AnthropicProtocolError(f"unsupported tool_choice type: {choice_type!r}")


def anthropic_to_openai_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one Anthropic Messages request to OpenAI chat-completion form."""
    request = _require_mapping(request, "request")
    model = _require_nonempty_string(request.get("model"), "model")
    if isinstance(max_tokens := request.get("max_tokens"), bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise AnthropicProtocolError("max_tokens must be a positive integer")
    if not isinstance(raw_messages := request.get("messages"), list):
        raise AnthropicProtocolError("messages must be an array")

    messages: list[dict[str, Any]] = []
    if (system := _convert_system(request.get("system"))) is not None:
        messages.append({"role": "system", "content": system})

    for message_index, raw_message in enumerate(raw_messages):
        message = _require_mapping(raw_message, f"messages[{message_index}]")
        if (role := message.get("role")) not in ("user", "assistant", "system"):
            raise AnthropicProtocolError(f"messages[{message_index}].role is invalid: {role!r}")
        _convert_message_content(messages, role, message.get("content"), message_index)

    converted: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": bool(request.get("stream", False)),
    }
    for source in ("temperature", "top_p", "top_k"):
        if (value := request.get(source)) is not None:
            converted[source] = value
    if (stop_sequences := request.get("stop_sequences")) is not None:
        converted["stop"] = stop_sequences

    if (tools := _convert_tools(request.get("tools"))) is not None:
        converted["tools"] = tools
    if (tool_choice := _convert_tool_choice(request.get("tool_choice"), tools)) is not None:
        converted["tool_choice"] = tool_choice
    return converted
