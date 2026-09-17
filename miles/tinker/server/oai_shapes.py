"""OpenAI ChatCompletion wire shapes for the recorded-session route: request body → TurnRequest, TurnResult → JSON."""

import time
import uuid
from typing import Any

from miles.tinker.core.tinker_session_server import TurnRequest, TurnResult
from miles.tinker.core.types import UserInputError


def _number(body: dict[str, Any], key: str, default: float) -> float:
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UserInputError(f"{key} must be a number")
    return value


def max_new_tokens_of(body: dict[str, Any]) -> int:
    """The body's max_tokens (or max_completion_tokens): a positive int, required as for Tinker sample."""
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if type(max_tokens) is not int or max_tokens < 1:
        raise UserInputError("max_tokens must be a positive integer (required, as for Tinker sample)")
    return max_tokens


def parse_chat_request(body: dict[str, Any]) -> TurnRequest:
    """OpenAI chat body → TurnRequest: n=1 only, no stream, max_tokens required, stop normalized to a list."""
    max_tokens = max_new_tokens_of(body)
    if body.get("n", 1) != 1:
        raise UserInputError("a recorded session samples one completion per turn; use n=1")
    if body.get("stream"):
        raise UserInputError("stream=true is not supported on the recorded session route")
    sampling_params: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": _number(body, "temperature", 1.0),
        "top_p": _number(body, "top_p", 1.0),
    }
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
    return TurnRequest(
        messages=body.get("messages"),
        tools=body.get("tools") or None,
        sampling_params=sampling_params,
        chat_template_kwargs=override,
        model=body.get("model"),
    )


def chat_completion_json(body: dict[str, Any], result: TurnResult) -> dict[str, Any]:
    """TurnResult → OpenAI ChatCompletion JSON: one choice with message/finish_reason plus usage."""
    prompt_tokens = len(result.turn.input_ids)
    completion_tokens = len(result.turn.output_ids)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.turn.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
