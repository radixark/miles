"""Bounded agentic pilot with explicit invalid-trial metadata."""
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import harbor_agent_function
from reward_policy import apply_reward_policy

_SEMAPHORE: asyncio.Semaphore | None = None

async def run(base_url: str, prompt: Any, request_kwargs: dict | None = None,
              metadata: dict | None = None, **kwargs: Any) -> dict:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(int(os.environ.get("SANDBOX_CONCURRENCY", "128")))
    request = dict(request_kwargs or {})
    request["timeout"] = 10800
    async with _SEMAPHORE:
        result = await harbor_agent_function.run(
            base_url=base_url, prompt=prompt, request_kwargs=request, metadata=metadata, **kwargs)
    result = apply_reward_policy(result)
    event = {"task": (metadata or {}).get("instance_id"), **result}
    with (Path(os.environ["PILOT_ROOT"]) / "pilot-verdicts.jsonl").open("a") as stream:
        stream.write(json.dumps(event, default=str) + "\n")
    return result
