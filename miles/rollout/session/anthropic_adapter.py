"""Anthropic protocol helpers for the session HTTP adapter."""

import json

from pydantic import ValidationError
from sglang.srt.entrypoints.anthropic import utils as anthropic_utils
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest, is_server_tool
from starlette.responses import Response

from miles.rollout.session.core import JSON_MEDIA_TYPE, _render_json

# Preserve end-to-end error metadata; drop headers tied to the replaced body.
_ANTHROPIC_ERROR_HEADER_ALLOWLIST = ("www-authenticate", "retry-after", "x-request-id")
_ANTHROPIC_ERROR_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")


def _anthropic_wire_json(model) -> bytes:
    return _render_json(model.model_dump(mode="json", exclude_none=True, by_alias=True))


def _anthropic_error_response(status_code: int, body: bytes, headers: dict | None = None) -> Response:
    envelope = anthropic_utils.to_anthropic_error(status_code, body)
    kept_headers = {
        k: v
        for k, v in (headers or {}).items()
        if k.lower() in _ANTHROPIC_ERROR_HEADER_ALLOWLIST or k.lower().startswith(_ANTHROPIC_ERROR_HEADER_PREFIXES)
    }
    return Response(
        content=_anthropic_wire_json(envelope),
        status_code=status_code,
        headers=kept_headers,
        media_type=JSON_MEDIA_TYPE,
    )


def _anthropic_sse_body(events) -> bytes:
    return b"".join(
        f"event: {event.type}\ndata: ".encode() + _anthropic_wire_json(event) + b"\n\n" for event in events
    )


def _parse_anthropic_request(body: bytes) -> AnthropicMessagesRequest:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    try:
        return AnthropicMessagesRequest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _validate_anthropic_content_block(block, *, allow_thinking: bool = False) -> None:
    if block.type == "thinking":
        if allow_thinking:
            return
        raise ValueError("thinking content blocks are only supported in assistant history")
    if block.type == "redacted_thinking":
        raise ValueError("redacted_thinking content blocks are not supported by this endpoint")
    if block.type == "image":
        raise ValueError("image content blocks are not enabled for this deployment")
    if block.type == "tool_reference":
        raise ValueError("tool_reference content blocks are not enabled for this deployment")
    if block.type == "search_result":
        raise ValueError("search_result content blocks are not enabled for this deployment")
    if block.type == "tool_result" and block.is_error is True:
        raise ValueError("tool_result is_error=true is not supported by this endpoint")


def _validate_anthropic_features(request: AnthropicMessagesRequest) -> None:
    if request.thinking is not None:
        raise ValueError("thinking is not supported by this endpoint")
    if request.output_config is not None:
        raise ValueError("output_config is not enabled for this deployment")
    if request.betas:
        raise ValueError("betas is not enabled for this deployment")
    if request.tools:
        for tool in request.tools:
            if is_server_tool(tool):
                raise ValueError(f"server tool {tool.name!r} (type={tool.type!r}) is not enabled for this deployment")

    if request.system is not None and not isinstance(request.system, str):
        for block in request.system:
            _validate_anthropic_content_block(block)
    for message in request.messages:
        if isinstance(message.content, str):
            continue
        for block in message.content:
            _validate_anthropic_content_block(block, allow_thinking=message.role == "assistant")
            if block.type == "tool_result" and isinstance(block.content, list):
                for nested_block in block.content:
                    _validate_anthropic_content_block(nested_block)


def _strip_anthropic_reasoning_history(
    anthropic_request: AnthropicMessagesRequest,
) -> tuple[AnthropicMessagesRequest, list[str | None]]:
    """Return a conversion copy plus canonical assistant reasoning history."""
    conversion_messages = []
    reasoning_history: list[str | None] = []
    for message in anthropic_request.messages:
        if message.role != "assistant":
            conversion_messages.append(message)
            continue
        if isinstance(message.content, str):
            conversion_messages.append(message)
            reasoning_history.append(None)
            continue
        thinking_blocks = [block for block in message.content if block.type == "thinking"]
        thinking_parts = [block.thinking for block in thinking_blocks if block.thinking]
        reasoning_history.append("\n".join(thinking_parts) or None)
        if thinking_blocks:
            message = message.model_copy(
                update={"content": [block for block in message.content if block.type != "thinking"]}
            )
        conversion_messages.append(message)
    return anthropic_request.model_copy(update={"messages": conversion_messages}), reasoning_history


def _restore_anthropic_reasoning_history(openai_body: dict, reasoning_history: list[str | None]) -> None:
    """Map replayed assistant thinking blocks back to canonical reasoning content."""
    assistants = [message for message in openai_body["messages"] if message["role"] == "assistant"]
    if len(assistants) != len(reasoning_history):
        raise ValueError(
            f"assistant history count changed during Anthropic conversion: "
            f"{len(reasoning_history)} before, {len(assistants)} after"
        )
    for message, reasoning_content in zip(assistants, reasoning_history, strict=True):
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
