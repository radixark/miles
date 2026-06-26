"""Phase-2 OpenEnv Coding adapter (HuggingFace OpenEnv <-> miles).

Same shape as ``openenv_echo_agent_function.run`` but drives the OpenEnv Coding
env (``PythonCodeActEnv``) instead of Echo, so the reward is a *real* learning
signal rather than a constant:

  1. Ask the *trained policy* (served by miles, reachable at ``base_url/v1``) to
     write a short Python snippet. Tokens flow through the session server, so
     miles captures token ids + logprobs + loss masks natively.
  2. Strip any markdown code fences from the model's reply.
  3. reset()+step() the Coding env with ``CodeAction(code=...)``; the env's
     safe-coding transform scores it (-1.0 dangerous import, -0.2 syntax error,
     +0.1 concise) and puts the reward on the step result.
  4. Return env metadata; ``openenv_generate.reward_func`` reads
     metadata["reward"] into the GRPO sample.

raw_reward should climb as the policy learns to emit short, safe, syntactically
valid Python -- this exercises GRPO learning end to end through the seam.

Env vars:
  OPENENV_ENV_URL       base_url of the OpenEnv Coding env server
  AGENT_MODEL_NAME      model name to send to the policy (default: "model")
  MILES_ROUTER_EXTERNAL_HOST  optional host rewrite for off-cluster agents
"""

import asyncio
import logging
import os
import random
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_DEFAULT_CODING_URL = "http://localhost:8002"

# A self-hosted concurrent coding server (serve_coding_concurrent.py) admits many
# sessions, but retry on transient CAPACITY_REACHED in case it is run single-slot.
_CAPACITY_MAX_WAIT_S = 180.0
_CAPACITY_BACKOFF_S = (0.25, 1.5)

# Strip a single fenced block: ```python\n...\n``` or ```\n...\n```.
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _resolve_session_url(base_url: str) -> str:
    """Build the OpenAI-compatible policy URL, rewriting host for off-cluster agents."""
    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        netloc = f"{external_host}:{parsed.port}" if parsed.port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))
    return session_url


def _extract_messages(prompt: Any) -> list[dict[str, str]]:
    """Accept either a chat-message list or a raw string prompt."""
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": str(prompt)}]


def _extract_code(text: str) -> str:
    """Pull Python source out of the model's reply, stripping markdown fences."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def _run_episode(CodingEnv: Any, CodeAction: Any, env_url: str, code: str) -> float:
    """reset()+step() the Coding env, retrying while a session slot is busy."""
    deadline = asyncio.get_event_loop().time() + _CAPACITY_MAX_WAIT_S
    while True:
        try:
            async with CodingEnv(base_url=env_url) as env:
                await env.reset()
                result = await env.step(CodeAction(code=code))
                return float(getattr(result, "reward", 0.0) or 0.0)
        except Exception as e:
            if "CAPACITY_REACHED" in str(e) and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(random.uniform(*_CAPACITY_BACKOFF_S))
                continue
            raise


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one OpenEnv Coding episode driven by the trained policy."""
    # Imported lazily so the file is importable without the env client installed.
    from coding_env import CodeAction, CodingEnv

    request_kwargs = request_kwargs or {}
    metadata = metadata or {}

    session_url = _resolve_session_url(base_url)
    model_name = os.getenv("AGENT_MODEL_NAME", os.getenv("SWE_AGENT_MODEL_NAME", "model"))
    env_url = os.getenv("OPENENV_ENV_URL", _DEFAULT_CODING_URL)

    policy = AsyncOpenAI(base_url=session_url, api_key="EMPTY")

    try:
        # Generate the policy action FIRST, with no env session held -- so the
        # (slow) LLM calls for every episode run concurrently through sglang.
        messages = _extract_messages(prompt)
        completion = await policy.chat.completions.create(
            model=model_name,
            messages=messages,
            extra_body=request_kwargs,
        )
        action_text = completion.choices[0].message.content or ""
        code = _extract_code(action_text)

        reward = await _run_episode(CodingEnv, CodeAction, env_url, code)
    except Exception as e:
        logger.error(f"OpenEnv Coding episode failed: {e}", exc_info=True)
        return None

    return {
        "reward": reward,
        "exit_status": "completed",
        "eval_report": {},
        "agent_metrics": {"turns": 1, "tool_calls": 1},
    }
