import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RolloutRequestContext:
    session_id: str


RolloutRequestHook = Callable[
    [dict[str, Any], RolloutRequestContext, dict[str, Any]],
    dict[str, Any] | Awaitable[dict[str, Any] | None] | None,
]


async def prepare_rollout_request(
    hook: RolloutRequestHook,
    hook_args: dict[str, Any],
    context: RolloutRequestContext,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    request = {
        "payload": payload,
        "headers": headers,
        "max_attempts": 1,
        "retry_interval": 1.0,
    }
    result = hook(hook_args, context, request)
    if inspect.isawaitable(result):
        result = await result
    if result is not None:
        if not isinstance(result, dict):
            raise TypeError(f"rollout request hook must return None or a dict, got {type(result).__name__}")
        request.update(result)

    request["max_attempts"] = int(request["max_attempts"])
    request["retry_interval"] = float(request["retry_interval"])
    if request["max_attempts"] < 1:
        raise ValueError("max_attempts must be at least 1")
    if request["retry_interval"] < 0:
        raise ValueError("retry_interval must be non-negative")
    if not isinstance(request["payload"], dict):
        raise TypeError("payload must be a dict")
    if not isinstance(request["headers"], dict):
        raise TypeError("headers must be a dict")
    return request
