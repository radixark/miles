"""Qwen 3.5 TITO adapter for one chess rollout."""

from collections.abc import Mapping
from typing import Any

from chess_eval.tito_v2 import run as run_chess

_REQUIRED_CHAT_TEMPLATE_KWARGS: dict[str, bool] = {
    "clear_thinking": False,
    "enable_thinking": True,
}


def _request_with_thinking(request_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    request = dict(request_kwargs or {})
    existing = request.get("chat_template_kwargs", {})
    if not isinstance(existing, Mapping):
        raise TypeError("request_kwargs.chat_template_kwargs must be a mapping")

    merged = dict(existing)
    for key, required_value in _REQUIRED_CHAT_TEMPLATE_KWARGS.items():
        if key in merged and merged[key] != required_value:
            raise ValueError(f"chat_template_kwargs.{key} must be {required_value!r}")
        merged[key] = required_value
    request["chat_template_kwargs"] = merged
    return request


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one game while preserving Qwen thinking in the TITO conversation."""

    return await run_chess(
        base_url=base_url,
        prompt=prompt,
        request_kwargs=_request_with_thinking(request_kwargs),
        metadata=dict(metadata or {}),
        **kwargs,
    )
