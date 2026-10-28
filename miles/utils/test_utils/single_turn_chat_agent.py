"""Single-turn chat agent for session-server e2e tests.

Sends the prompt messages as one non-streaming chat completion on the session base URL, so a plain
single-turn dataset trains through the session server (``--custom-agent-function-path
miles.utils.test_utils.single_turn_chat_agent.run_agent``).
"""

from typing import Any

import httpx

# Arbitrary upper bound; a stuck request fails the rollout instead of hanging until the CI job timeout.
_REQUEST_TIMEOUT_SECONDS = 600


async def run_agent(
    base_url: str, prompt: list[dict[str, Any]] | str, request_kwargs: dict[str, Any], metadata: dict, **kwargs
) -> None:
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    payload = {**request_kwargs, "model": "default", "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{base_url}/v1/chat/completions", json=payload)
    response.raise_for_status()
