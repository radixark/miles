"""Grade single-turn Sokoban completions with NeMo Gym's real /verify endpoint."""

import asyncio
import math
import os
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from miles.utils.types import Sample


def build_verify_request(sample: "Sample") -> dict[str, Any]:
    """Keep the reference answer in the verifier payload, outside the model prompt."""
    question = sample.metadata["sokoban_question"]
    return {
        "responses_create_params": {"input": [{"role": "user", "content": question}]},
        "question": question,
        "answer": sample.label,
        "metadata": sample.metadata,
        "response": {
            "id": "sokoban-response",
            "created_at": 0,
            "model": "policy",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": [
                {
                    "id": "sokoban-message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": sample.response, "annotations": []}],
                }
            ],
        },
    }


async def _score(client: httpx.AsyncClient, url: str, sample: "Sample") -> float:
    response = await client.post(f"{url.rstrip('/')}/verify", json=build_verify_request(sample))
    response.raise_for_status()
    result = response.json()
    if result.get("task_name") != "sokoban" or result.get("mask_sample") or result.get("failure_reason"):
        raise RuntimeError(f"NeMo Gym returned an invalid Sokoban grade: {result}")
    reward = float(result["reward"])
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise ValueError(f"Expected a binary Sokoban reward, got {reward}")
    sample.metadata["sokoban_extracted_answer"] = result["extracted_answer"]
    sample.metadata["sokoban_reward"] = reward
    return reward


async def reward_func(args: Any, samples: "Sample | list[Sample]", **kwargs: Any) -> float | list[float]:
    """Support Miles' individual and batched reward calls; HTTP failures fail loudly."""
    url = os.environ["NEMO_GYM_SOKOBAN_URL"]
    batch = samples if isinstance(samples, list) else [samples]
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        rewards = await asyncio.gather(*(_score(client, url, sample) for sample in batch))
    return rewards if isinstance(samples, list) else rewards[0]
