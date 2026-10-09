"""OpenAI ChatCompletion wire shapes for the recorded-session route: request body → TurnRequest, TurnResult → JSON."""

import time
import uuid
from typing import Any

from miles.tinker.core.tinker_session_server import TurnRequest, TurnResult, max_new_tokens_of
from miles.tinker.core.types import UserInputError


def _number(body: dict[str, Any], key: str) -> float:
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UserInputError(f"{key} must be a number")
    return value


# OpenAI parameters that change what is generated and have no Tinker sampling equivalent, with the values that
# mean "off" in OpenAI's own schema; anything else is refused, never ignored
_UNSUPPORTED_SAMPLING: dict[str, tuple[Any, ...]] = {
    "presence_penalty": (None, 0),
    "frequency_penalty": (None, 0),
    "logit_bias": (None, {}),
    "logprobs": (None, False),
    "top_logprobs": (None, 0),
    "reasoning_effort": (None,),
    "separate_reasoning": (None, False),  # the route never splits reasoning out of content
}


def _refuse_unsupported(body: dict[str, Any]) -> None:
    refused = [key for key, off in _UNSUPPORTED_SAMPLING.items() if body.get(key) not in off]
    if body.get("tool_choice") not in (None, "auto"):
        refused.append("tool_choice")  # tools are rendered into the prompt; the model chooses freely
    if body.get("response_format") not in (None, {"type": "text"}):
        refused.append("response_format")
    if body.get("parallel_tool_calls") is False:
        refused.append("parallel_tool_calls")
    if refused:
        raise UserInputError(f"not supported on the recorded session route: {', '.join(refused)}")


def parse_chat_request(body: dict[str, Any]) -> TurnRequest:
    """OpenAI chat body → TurnRequest: n=1 only, no stream, max_tokens required, stop normalized to a list."""
    sampling_params: dict[str, Any] = {"max_tokens": body.get("max_tokens", body.get("max_completion_tokens"))}
    max_new_tokens_of(sampling_params)
    if body.get("n", 1) != 1:
        raise UserInputError("a recorded session samples one completion per turn; use n=1")
    if body.get("stream"):
        raise UserInputError("stream=true is not supported on the recorded session route")
    _refuse_unsupported(body)
    for key in ("temperature", "top_p"):  # omitted: runtime's defaults
        if key in body:
            sampling_params[key] = _number(body, key)
    for key in ("top_k", "seed"):
        if body.get(key) is not None:
            sampling_params[key] = body[key]
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop:
        if not isinstance(stop, list) or not all(isinstance(item, str) for item in stop):
            raise UserInputError("stop must be a string or a list of strings")
        sampling_params["stop"] = list(stop)
    override = body.get("chat_template_kwargs")
    if override is not None and not isinstance(override, dict):
        raise UserInputError("chat_template_kwargs must be an object")
    if body.get("model") is not None and not isinstance(body["model"], str):
        raise UserInputError("model must be a string")
    return TurnRequest(
        messages=body.get("messages"),
        tools=body.get("tools") or None,
        sampling_params=sampling_params,
        chat_template_kwargs=override,
        model=body.get("model"),
    )


def chat_completion_json(body: dict[str, Any], result: TurnResult) -> dict[str, Any]:
    """TurnResult → OpenAI ChatCompletion JSON: one choice (assistant message, finish_reason) plus usage."""
    prompt_tokens = len(result.turn.input_ids)
    completion_tokens = len(result.turn.output_ids)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": dict(result.assistant_message),
                "finish_reason": result.turn.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
