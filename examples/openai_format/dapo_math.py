"""
Custom agent example: single-turn DAPO math via OpenAI endpoints.
"""

from __future__ import annotations

from typing import Any


# Notice: only function based agent can use post API in miles
from miles.utils.http_utils import post


async def run_agent(
    base_url: str,
    prompt: list[dict[str, Any]] | str,
    tools: list[dict[str, Any]] | None = None,
    completion_params: dict[str, Any] | None = None,
) -> None:
    payload = {"model": "default", "messages": prompt, "tools": tools or [], "logprobs": True, **(completion_params or {})}
    await post(base_url + "/v1/chat/completions", payload)
