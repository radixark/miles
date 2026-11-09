"""Pydantic AI agent function for ``agentic_tool_call.generate``.

Runs one arithmetic task per call: the agent calls ``calculate`` through the
session URL, and the verdict compares the last number it writes with
``metadata["answer"]``.

Env vars (read on the rollout worker):
  PYDANTIC_AI_MAX_MODEL_CALLS    model calls per episode (default 8)
  PYDANTIC_AI_EPISODE_TIMEOUT_S  wall-clock cap per episode (default 600)
"""

import asyncio
import operator
import os
import re
from typing import Any, Literal

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import AgentRunError, ModelAPIError, ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from miles.rollout.agentic.agent_function import InfraAbort

INSTRUCTIONS = (
    "Use the calculate tool for every arithmetic step. "
    "When you are done, reply with the final result as a plain number."
)
_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def calculate(a: float, op: Literal["+", "-", "*", "/"], b: float) -> float:
    """Return `a op b`."""
    if op == "/" and b == 0:
        raise ModelRetry("division by zero")
    return _OPERATORS[op](a, b)


async def run(base_url: str, prompt: list[dict[str, Any]] | str, metadata: dict, **kwargs) -> dict:
    model = OpenAIChatModel("default", provider=OpenAIProvider(base_url=f"{base_url}/v1", api_key="sk-dummy"))
    agent = Agent(model, instructions=INSTRUCTIONS, tools=[calculate])
    limits = UsageLimits(request_limit=int(os.environ.get("PYDANTIC_AI_MAX_MODEL_CALLS", "8")))
    timeout = float(os.environ.get("PYDANTIC_AI_EPISODE_TIMEOUT_S", "600"))
    try:
        result = await asyncio.wait_for(agent.run(_user_prompt(prompt), usage_limits=limits), timeout)
    except ModelHTTPError as e:
        if e.status_code >= 500:
            raise InfraAbort("ModelServerError", str(e)) from e
        return {"reward": 0.0, "exit_status": f"ModelHTTP{e.status_code}"}
    except ModelAPIError as e:
        raise InfraAbort("ModelUnreachable", str(e)) from e
    except (AgentRunError, TimeoutError) as e:
        return {"reward": 0.0, "exit_status": type(e).__name__}
    answer = _last_number(result.output)
    correct = answer is not None and abs(answer - float(metadata["answer"])) < 1e-6
    return {"reward": float(correct), "exit_status": "Completed", "answer": answer}


def _user_prompt(prompt: list[dict[str, Any]] | str) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n\n".join(message["content"] for message in prompt if message["role"] == "user")


def _last_number(text: str) -> float | None:
    numbers = _NUMBER.findall(text)
    return float(numbers[-1].replace(",", "")) if numbers else None
