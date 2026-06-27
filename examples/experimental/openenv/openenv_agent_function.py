"""Generic OpenEnv <-> miles adapter: one agent function for every OpenEnv env.

miles selects the policy and calls ``run`` once per episode via
``--custom-agent-function-path openenv_agent_function.run``. Which OpenEnv
environment that episode drives is chosen by ``OPENENV_ENV_TYPE``
(echo | coding | tbench2 | swe), so the same file serves all tasks -- only the
prompt-data and the env server differ. There is no per-task (or per-env) agent
function to write: new single-turn envs are one entry in ``_ENV_SPECS``.

The policy is always reached at ``base_url/v1`` through miles' session server,
so token ids + logprobs + loss masks are captured natively (no re-tokenization)
on every turn, including each turn of a multi-turn env.

Per-env behavior lives in ``_ENV_SPECS``:
  * ``loader``      -- lazy import of the env client + action class,
  * ``default_url`` -- env server URL when OPENENV_ENV_URL is unset,
  * ``multi_turn``  -- single ``step`` (echo/coding) vs an agentic loop (tbench2/swe),
  * ``build_action``-- (single-turn only) policy text -> env action.

Env vars:
  OPENENV_ENV_TYPE   echo | coding | tbench2 | swe   (required)
  OPENENV_ENV_URL    base_url of the env server (default: per-env)
  OPENENV_MAX_TURNS  multi-turn cap (default: 30)
  OPENENV_MESSAGE_TIMEOUT_S  per-message WS recv timeout (default: 600; docker-mode
                     reset/exec/pytest routinely exceed the client default of 60)
  AGENT_MODEL_NAME   model name sent to the policy (default: "model")
  MILES_ROUTER_EXTERNAL_HOST  optional host rewrite for off-cluster agents
"""

import asyncio
import logging
import os
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Env slots are finite: hosted spaces admit one session at a time
# (CAPACITY_REACHED) and the docker-mode tbench2 server caps concurrent envs
# (MAX_CONCURRENT_ENVS), closing the WebSocket cleanly (ConnectionClosedOK) once
# full. Both are transient -- episodes hold a slot only for their rollout -- so
# jittered backoff + retry serializes the surplus rather than failing it.
#
# A rollout fans out more episodes than the server has slots (e.g. ~32 episodes
# vs a 16-session cap), so a queued episode must outwait a full episode ahead of
# it -- minutes, not seconds. The wait deadline is sized for that; the backoff
# ceiling is wide enough that 16 queued episodes don't hammer the server with
# reconnects while they wait.
_CAPACITY_MAX_WAIT_S = 1800.0
_CAPACITY_BACKOFF_S = (1.0, 5.0)

# Strip a single fenced block: ```python / ```bash / ``` ... ```.
_FENCE_RE = re.compile(r"```(?:python|py|bash|sh)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

# Max chars of command output fed back to the policy per turn (keeps context bounded).
_OBS_CHAR_CAP = 4000

# Per-message WS recv timeout. Docker-mode tbench2 reset (container create),
# exec, and evaluate (pytest) each routinely exceed the EnvClient default of 60s.
_MESSAGE_TIMEOUT_S = float(os.getenv("OPENENV_MESSAGE_TIMEOUT_S", "600"))


def _is_retryable_env_error(e: BaseException) -> bool:
    """True when an env op failed only because no env slot was free (transient).

    The docker-mode tbench2 server caps concurrent envs; over that cap it either
    returns CAPACITY_REACHED or closes the WebSocket cleanly (ConnectionClosedOK).
    Both mean "retry once a slot frees up", not a genuine episode failure. Match
    the close exceptions by class name so the adapter need not import websockets.
    """
    if "CAPACITY_REACHED" in str(e):
        return True
    return type(e).__name__ in {"ConnectionClosedOK", "ConnectionClosedError", "ConnectionClosed"}


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
        return list(prompt)
    return [{"role": "user", "content": str(prompt)}]


def _strip_fence(text: str) -> str:
    """Return the contents of a single fenced block, else the stripped text."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _obs_field(result: Any, name: str) -> str:
    """Read an Observation field off a StepResult, tolerating shape differences."""
    obs = getattr(result, "observation", result)
    return str(getattr(obs, name, "") or "")


# ---- per-env loaders (lazy so the file imports without every env client) ----


def _load_echo() -> dict[str, Any]:
    from echo_env import CallToolAction, EchoEnv

    return {"env": EchoEnv, "action": CallToolAction}


def _load_coding() -> dict[str, Any]:
    from coding_env import CodeAction, CodingEnv

    return {"env": CodingEnv, "action": CodeAction}


def _load_tbench2() -> dict[str, Any]:
    from tbench2_env import Tbench2Action, Tbench2Env

    return {"env": Tbench2Env, "action": Tbench2Action}


def _load_swe() -> dict[str, Any]:
    from swe_env import SweAction, SweEnv

    return {"env": SweEnv, "action": SweAction}


def _echo_action(classes: dict[str, Any], text: str) -> Any:
    return classes["action"](tool_name="echo_message", arguments={"message": text})


def _coding_action(classes: dict[str, Any], text: str) -> Any:
    return classes["action"](code=_strip_fence(text))


@dataclass
class EnvSpec:
    loader: Callable[[], dict[str, Any]]
    default_url: str
    multi_turn: bool
    build_action: Callable[[dict[str, Any], str], Any] | None = None


_ENV_SPECS: dict[str, EnvSpec] = {
    "echo": EnvSpec(_load_echo, "https://openenv-echo-env.hf.space", False, _echo_action),
    "coding": EnvSpec(_load_coding, "http://localhost:8002", False, _coding_action),
    "tbench2": EnvSpec(_load_tbench2, "http://localhost:8003", True),
    "swe": EnvSpec(_load_swe, "http://localhost:8004", True),
}


async def _with_env(env_cls: Any, env_url: str, body: Callable[[Any], Any]) -> Any:
    """Open an env session and run ``body(env)``, retrying while a slot is busy."""
    deadline = asyncio.get_event_loop().time() + _CAPACITY_MAX_WAIT_S
    while True:
        try:
            async with env_cls(base_url=env_url, message_timeout_s=_MESSAGE_TIMEOUT_S) as env:
                return await body(env)
        except Exception as e:
            if _is_retryable_env_error(e) and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(random.uniform(*_CAPACITY_BACKOFF_S))
                continue
            raise


async def _single_turn(
    spec: EnvSpec,
    classes: dict[str, Any],
    env_url: str,
    policy: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float, dict[str, int]]:
    """One policy call -> one env step -> reward off the step result (echo, coding)."""
    completion = await policy.chat.completions.create(
        model=model_name, messages=messages, extra_body=request_kwargs
    )
    text = completion.choices[0].message.content or ""
    action = spec.build_action(classes, text)

    async def body(env: Any) -> float:
        await env.reset()
        result = await env.step(action)
        return float(getattr(result, "reward", 0.0) or 0.0)

    reward = await _with_env(classes["env"], env_url, body)
    return reward, {"turns": 1, "tool_calls": 1}


async def _multi_turn(
    spec: EnvSpec,
    classes: dict[str, Any],
    env_url: str,
    policy: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float, dict[str, int]]:
    """Agentic loop: reset(task) -> {policy -> exec -> feed output back} -> evaluate (tbench2).

    The policy emits one shell command per turn (a ```bash block or the bare
    reply); the loop ends when the policy stops emitting a command, says
    TASK_COMPLETE, or hits OPENENV_MAX_TURNS. The final ``evaluate`` action runs
    the task's pytest suite and returns the binary reward.
    """
    action_cls = classes["action"]
    task_id = metadata.get("task_id") or metadata.get("task_name")
    max_turns = int(os.getenv("OPENENV_MAX_TURNS", "30"))

    async def body(env: Any) -> tuple[float, int]:
        reset_result = await (env.reset(task_id=task_id) if task_id else env.reset())
        instruction = _obs_field(reset_result, "instruction")
        convo = list(messages)
        if instruction:
            convo.append({"role": "user", "content": instruction})

        turns = 0
        while turns < max_turns:
            turns += 1
            completion = await policy.chat.completions.create(
                model=model_name, messages=convo, extra_body=request_kwargs
            )
            message = completion.choices[0].message
            reply = message.content or ""
            # Echo the assistant turn back verbatim. The session server stores the
            # message exactly as SGLang emitted it -- content plus reasoning_content
            # and tool_calls split out by the reasoning/tool-call parsers -- and
            # matches each later request against that stored prefix. A thin
            # {role, content} dict drops reasoning_content/tool_calls, diverges at
            # the assistant turn, and trips "rollback failed: no assistant message
            # in matched prefix". model_dump round-trips whatever the SDK parsed
            # (extras like reasoning_content included).
            convo.append(message.model_dump(exclude_none=True))

            command = _strip_fence(reply) if "```" in reply else reply.strip()
            if not command or command.upper().startswith("TASK_COMPLETE"):
                break

            step_result = await env.step(action_cls(action_type="exec", command=command))
            output = _obs_field(step_result, "output")
            # Feed the command output back as a user turn, not a tool turn. GLM
            # emits native tool_calls that we must echo verbatim (above) for the
            # session server's prefix match; a role="tool" reply would then have
            # to carry a matching tool_call_id and trips OpenAI tool-call
            # validation. A plain user turn sidesteps the handshake -- the same
            # text protocol the Harbor mini-swe-agent scaffold uses.
            #
            # Substitute a placeholder when a command produces no stdout: SGLang
            # rejects an empty message content with "content cannot be empty".
            content = output[:_OBS_CHAR_CAP] or "(no output)"
            convo.append({"role": "user", "content": content})

        eval_result = await env.step(action_cls(action_type="evaluate"))
        reward = float(getattr(eval_result, "reward", 0.0) or 0.0)
        return reward, turns

    reward, turns = await _with_env(classes["env"], env_url, body)
    return reward, {"turns": turns, "tool_calls": turns}


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one OpenEnv episode (env chosen by OPENENV_ENV_TYPE) via the trained policy."""
    request_kwargs = request_kwargs or {}
    metadata = metadata or {}

    env_type = os.getenv("OPENENV_ENV_TYPE", "").strip().lower()
    spec = _ENV_SPECS.get(env_type)
    if spec is None:
        raise ValueError(
            f"OPENENV_ENV_TYPE must be one of {sorted(_ENV_SPECS)}; got {env_type!r}"
        )

    classes = spec.loader()
    session_url = _resolve_session_url(base_url)
    model_name = os.getenv("AGENT_MODEL_NAME", os.getenv("SWE_AGENT_MODEL_NAME", "model"))
    env_url = os.getenv("OPENENV_ENV_URL", spec.default_url)

    policy = AsyncOpenAI(base_url=session_url, api_key="EMPTY")
    messages = _extract_messages(prompt)
    runner = _multi_turn if spec.multi_turn else _single_turn

    try:
        reward, agent_metrics = await runner(
            spec, classes, env_url, policy, model_name, messages, request_kwargs, metadata
        )
    except Exception as e:
        logger.error(f"OpenEnv {env_type} episode failed: {e}", exc_info=True)
        return None

    return {
        "reward": reward,
        "exit_status": "completed",
        "eval_report": {},
        "agent_metrics": agent_metrics,
    }
