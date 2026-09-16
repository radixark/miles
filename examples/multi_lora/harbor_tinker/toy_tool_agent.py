"""Sandbox-free agent for stage 0: a few tool-calling turns against a recorded gateway session.

Same contract as ``examples/experimental/harbor/harbor_agent_function.run`` so the loop can swap the two.
Skeleton: functions document what they will do; bodies land in follow-up commits.
"""

from __future__ import annotations

from typing import Any


def calculator(expression: str) -> str:
    """Evaluate a small arithmetic expression locally; the only tool the toy agent offers."""
    raise NotImplementedError


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Drive two or three chat turns (tool call → tool result → final answer) against {base_url}/v1/chat/completions and return {reward, exit_status} by exact match with metadata['answer']."""
    raise NotImplementedError
