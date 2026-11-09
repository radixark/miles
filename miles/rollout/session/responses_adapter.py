import json
import time

from starlette.responses import Response

try:
    from sglang.srt.entrypoints.openai.responses_adapters import (
        custom_tool_description,
        custom_tool_parameters,
        decode_custom_tool_input,
        decode_reasoning_state,
        encode_custom_tool_input,
        encode_reasoning_state,
        label_developer_content,
    )
except ImportError:
    custom_tool_description = None
    custom_tool_parameters = None
    decode_custom_tool_input = None
    decode_reasoning_state = None
    encode_custom_tool_input = None
    encode_reasoning_state = None
    label_developer_content = None

from miles.rollout.session.core import JSON_MEDIA_TYPE, _render_json

_RESPONSES_TEXT_PART_TYPES = ("input_text", "output_text", "text")


def responses_adapter_available() -> bool:
    return encode_reasoning_state is not None


def _responses_error_response(status_code: int, message: str) -> Response:
    error_type = "invalid_request_error" if status_code < 500 else "server_error"
    body = {"error": {"message": message, "type": error_type, "param": None, "code": None}}
    return Response(content=_render_json(body), status_code=status_code, media_type=JSON_MEDIA_TYPE)


def _responses_error_message(body: bytes) -> str:
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return body.decode("utf-8", "replace")
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)


def _responses_text(content) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    texts = []
    for part in content:
        if part.get("type") not in _RESPONSES_TEXT_PART_TYPES:
            raise ValueError(f"{part.get('type')} content parts are not enabled for this deployment")
        texts.append(part.get("text", ""))
    return "\n".join(texts)


def _responses_tools_to_chat(tools) -> tuple[list[dict], set[str]]:
    chat_tools, custom_names = [], set()
    for tool in tools or []:
        if tool.get("type") == "function":
            function = {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool}
        elif tool.get("type") == "custom" and tool.get("name"):
            custom_names.add(tool["name"])
            function = {
                "name": tool["name"],
                "description": custom_tool_description(tool.get("description"), tool.get("format")),
                "parameters": custom_tool_parameters(),
            }
        else:
            continue
        chat_tools.append({"type": "function", "function": function})
    return chat_tools, custom_names


def _responses_tool_choice(tool_choice):
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") in ("function", "custom"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return None


def _responses_reasoning_text(item: dict) -> str | None:
    restored = decode_reasoning_state(item.get("encrypted_content"))
    if restored:
        return restored
    for key in ("summary", "content"):
        texts = [part["text"] for part in item.get(key) or [] if isinstance(part, dict) and part.get("text")]
        if texts:
            return "\n".join(texts)
    return None


def _responses_item_to_message(item: dict) -> dict | None:
    kind = item.get("type", "message")
    if kind == "message":
        role = item.get("role")
        if role not in ("user", "assistant", "system", "developer"):
            raise ValueError(f"unsupported message role: {role!r}")
        content = _responses_text(item.get("content"))
        if role == "developer":
            return {"role": "system", "content": label_developer_content(content)}
        return {"role": role, "content": content}
    if kind in ("function_call", "custom_tool_call"):
        arguments = (
            item.get("arguments") if kind == "function_call" else encode_custom_tool_input(item.get("input", ""))
        )
        call = {"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": arguments}}
        return {"role": "assistant", "tool_calls": [call]}
    if kind in ("function_call_output", "custom_tool_call_output"):
        return {"role": "tool", "tool_call_id": item["call_id"], "content": _responses_text(item.get("output"))}
    if kind == "reasoning":
        text = _responses_reasoning_text(item)
        return {"role": "assistant", "reasoning_content": text} if text else None
    raise ValueError(f"unsupported input item type: {kind!r}")


def _merge_into_assistant(previous: dict, message: dict) -> None:
    content = message.get("content")
    if content:
        previous["content"] = f"{previous['content']}\n\n{content}" if previous.get("content") else content
    if message.get("tool_calls"):
        previous["tool_calls"] = previous.get("tool_calls", []) + message["tool_calls"]
    if message.get("reasoning_content"):
        previous["reasoning_content"] = message["reasoning_content"]


def _responses_input_to_messages(instructions, items, *, merge_inline_system: bool) -> list[dict]:
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    messages = [{"role": "system", "content": instructions}] if instructions else []
    for item in items or []:
        message = _responses_item_to_message(item)
        if message is None:
            continue
        if message["role"] == "assistant" and messages and messages[-1]["role"] == "assistant":
            _merge_into_assistant(messages[-1], message)
        else:
            messages.append(message)
    if not merge_inline_system:
        return messages
    system = [message["content"] for message in messages if message["role"] == "system"]
    rest = [message for message in messages if message["role"] != "system"]
    return ([{"role": "system", "content": "\n".join(system)}] if system else []) + rest


def _responses_to_chat_body(body: bytes, *, merge_inline_system: bool) -> tuple[dict, dict, set[str]]:
    try:
        request = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(request, dict):
        raise ValueError("request body must be a JSON object")
    if request.get("previous_response_id"):
        raise ValueError("previous_response_id is not supported: replay the full input instead")
    chat_tools, custom_names = _responses_tools_to_chat(request.get("tools"))
    chat = {key: request[key] for key in ("model", "temperature", "top_p") if request.get(key) is not None}
    chat["messages"] = _responses_input_to_messages(
        request.get("instructions"), request.get("input"), merge_inline_system=merge_inline_system
    )
    if request.get("max_output_tokens") is not None:
        chat["max_tokens"] = request["max_output_tokens"]
    if chat_tools:
        chat["tools"] = chat_tools
        tool_choice = _responses_tool_choice(request.get("tool_choice"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
        if request.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = request["parallel_tool_calls"]
    return request, chat, custom_names


def _chat_message_to_output_items(message: dict, response_id: str, custom_names: set[str]) -> list[dict]:
    items = []
    if message.get("reasoning_content"):
        text = message["reasoning_content"]
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{response_id}",
                "summary": [{"type": "summary_text", "text": text}],
                "encrypted_content": encode_reasoning_state(text),
            }
        )
    if message.get("content"):
        items.append(
            {
                "type": "message",
                "id": f"msg_{response_id}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": message["content"], "annotations": []}],
            }
        )
    for index, call in enumerate(message.get("tool_calls") or []):
        name, arguments = call["function"]["name"], call["function"]["arguments"]
        item = {"id": f"fc_{response_id}_{index}", "call_id": call["id"], "name": name, "status": "completed"}
        if name in custom_names:
            items.append({"type": "custom_tool_call", **item, "input": decode_custom_tool_input(arguments)})
        else:
            items.append({"type": "function_call", **item, "arguments": arguments})
    return items


def _chat_to_response(chat_response: dict, request: dict, custom_names: set[str]) -> dict:
    choice = chat_response["choices"][0]
    usage = chat_response.get("usage") or {}
    truncated = choice.get("finish_reason") == "length"
    return {
        "id": chat_response["id"],
        "object": "response",
        "created_at": chat_response.get("created", int(time.time())),
        "status": "incomplete" if truncated else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
        "error": None,
        "model": chat_response.get("model"),
        "output": _chat_message_to_output_items(choice["message"], chat_response["id"], custom_names),
        "instructions": request.get("instructions"),
        "max_output_tokens": request.get("max_output_tokens"),
        "metadata": request.get("metadata") or {},
        "parallel_tool_calls": request.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "store": False,
        "temperature": request.get("temperature"),
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools") or [],
        "top_p": request.get("top_p"),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _responses_sse_events(response: dict) -> list[dict]:
    started = {**response, "status": "in_progress", "output": [], "usage": None}
    events = [{"type": "response.created", "response": started}, {"type": "response.in_progress", "response": started}]
    for index, item in enumerate(response["output"]):
        events.append({"type": "response.output_item.added", "output_index": index, "item": item})
        if item["type"] == "message":
            text = item["content"][0]["text"]
            delta = {"output_index": index, "content_index": 0, "item_id": item["id"], "logprobs": []}
            events.append({"type": "response.output_text.delta", **delta, "delta": text})
            events.append({"type": "response.output_text.done", **delta, "text": text})
        elif item["type"] == "function_call":
            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": index,
                    "item_id": item["id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    final_type = "response.completed" if response["status"] == "completed" else "response.incomplete"
    events.append({"type": final_type, "response": response})
    return [{**event, "sequence_number": number} for number, event in enumerate(events)]


def _responses_sse_body(response: dict) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: ".encode() + _render_json(event) + b"\n\n"
        for event in _responses_sse_events(response)
    )
