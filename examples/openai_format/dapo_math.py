"""
Custom agent example: single-turn DAPO math via OpenAI endpoints.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

# Notice: only function based agent can use post API in miles
from miles.utils.http_utils import post


async def run_agent(base_url: str, prompt: list[dict[str, Any]] | str, sampling_params: dict[str, Any]) -> None:
    request_kwargs = _build_chat_request_kwargs(sampling_params)
    payload = {
        "model": "default",
        "messages": prompt,
        "logprobs": True,
        **request_kwargs,
    }
    await post(base_url + "/v1/chat/completions", payload)


# Process keys to match ChatCompletionRequest input
def _build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
