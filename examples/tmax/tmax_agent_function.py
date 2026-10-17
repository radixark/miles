"""TMax Terminal-RL agent loop for ``agentic_tool_call.generate``.

Miles calls ``run`` once per episode via ``--custom-agent-function-path
tmax_agent_function.run``. The policy is reached through the session server at
``base_url/v1``, so token ids, logprobs and loss masks are recorded server-side
on every turn; this module only owns the loop and the sandbox.

The loop follows open-instruct's Terminal-RL rollout (``vllm_utils.process_request``
with the ``swerl_vanillux_sandbox`` env, ``--tool_call_format_error_feedback`` off,
``SWERL_RESET_FAILURE_ZERO_REWARD=1``):

  start a fresh container for the task (5 attempts)
  repeat up to TMAX_MAX_STEPS tool calls:
      chat completion with the single `bash` tool
      no call to a known tool / length-truncated turn / context full  -> stop, reward 0
      run each bash call; a failed call becomes its observation         -> continue
      the submit marker runs the tests                                  -> reward, stop

Failure semantics:

* sandbox cannot be started after 5 attempts: with ``TMAX_START_FAILURE=zero`` (the
  default, as open-instruct) the episode scores 0 and stays in its group, through a
  minimal one-token turn so the session has a sample; ``drop`` makes no model call,
  so the sample is ABORTED and the whole group is dropped;
* a bash command or the tests failing for infra reasons: the error is the
  observation and the episode continues (a vanished container is recreated once);
* sandbox OOM, a remote sandbox the policy got removed (gateway backend: over its memory or process
  limit, or stopped), or the optional episode wall-clock cap: reward 0;
* a remote sandbox the fleet lost (gateway backend: its replica left, its binding expired, or it vanished
  for a reason the replica cannot name): ``reward=None``, so the missing-reward filter drops the group;
* over-sampling abort: ``reward=None``; the group is discarded anyway.

Each call runs in its own process under ``--custom-agent-function-mode subproc``
(the default), so the abort hook, which runs in the rollout process, reaches
in-flight episodes through a generation counter in a file scoped to that rollout
process, plus removal of the rollout's containers to cut long commands short.

Env vars:
  TMAX_TASK_DATA_DIR       directory holding one sub-directory per task_id (required).
  TMAX_MAX_STEPS           max bash calls per episode (default: 64).
  TMAX_START_FAILURE       zero | drop: what a sandbox that never starts does (default: zero).
  TMAX_EPISODE_TIMEOUT_S   wall-clock cap per episode, 0 for none (default: 0, as open-instruct).
  TMAX_TOOL_CALL_TIMEOUT_S outer cap on one step, a submit's test run included (default: 120, open-instruct's
                           tool_call_timeout); an overrun is an error observation, so it also bounds a hung podman.
  TMAX_TRACE_DIR           if set, one JSONL line per episode (messages, reward, metrics) is
                           appended under this directory (open-instruct's --save_traces).
  AGENT_MODEL_NAME         model name sent to the policy (default: "model").
  plus the sandbox knobs documented in tmax_sandbox.py.
"""

import asyncio
import contextlib
import json
import logging
import os
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tmax_gateway import TmaxGatewaySandbox, close_live_sandboxes
from tmax_sandbox import (
    SandboxBase,
    SandboxInfraError,
    SandboxLostError,
    SandboxOOMError,
    TmaxSandbox,
    backend_name,
    remove_owned_sandboxes,
)

from miles.rollout.agentic.rollout_limits import current_scope, rollout_lock
from miles.rollout.agentic.session import openai_session_url

logger = logging.getLogger(__name__)

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. "
            "Working directory and environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}

_FORMAT_ERROR = (
    "Format error: {error}\n\n"
    "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
    "end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    "via the `bash` tool, with no other content in the command.\n"
)

_ABORT_POLL_S = 1.0


# Losses the fleet caused rather than the policy; a lost episode scores 0 for any other reason (memory_limit,
# pids_limit, container_stopped: the agent can stop its own container).
_FLEET_LOSS_REASONS = frozenset({"unknown", "affinity_owner_lost", "binding_lost"})


def _lost_sandbox_reward(reason: str) -> float | None:
    """None (drop the group) for a loss the fleet caused, 0 for one the policy did."""
    return None if reason in _FLEET_LOSS_REASONS else 0.0


class _Aborted(Exception):
    """The rollout step aborted this episode (over-sampling); its group is discarded."""


def _abort_generation_file() -> Path:
    """Shared by the rollout process and its agent processes (they inherit its scope)."""
    return Path(tempfile.gettempdir()) / f"miles-tmax-abort-{current_scope()}"


def _abort_generation() -> int:
    try:
        return int(_abort_generation_file().read_text() or 0)
    except FileNotFoundError:
        return 0


@dataclass
class _Episode:
    task_id: str
    max_seq_len: int | None
    # The abort generation the episode started under; a later abort ends it, an earlier one does not.
    abort_generation: int
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    end_reason: str = "max_steps"
    # Why the gateway backend's fleet lost the container (end_reason sandbox_lost): memory_limit, pids_limit, ...
    sandbox_lost_reason: str = ""
    reward: float | None = 0.0
    last_total_tokens: int = 0
    messages: list[dict] = field(default_factory=list)
    gen_times: list[float] = field(default_factory=list)
    tool_times: list[float] = field(default_factory=list)
    # Seconds from asking for the sandbox to holding it (or to giving up): open-instruct's tools/env_reset runtime.
    sandbox_start_time: float = 0.0


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    metadata = metadata or {}
    task_id = metadata["task_id"]
    episode = _Episode(
        task_id=task_id,
        max_seq_len=metadata.get("max_seq_len"),
        abort_generation=_abort_generation(),
        messages=list(prompt),
    )
    policy = AsyncOpenAI(base_url=openai_session_url(base_url), api_key="EMPTY", timeout=3600)
    request_kwargs = dict(request_kwargs or {})
    t_start = time.monotonic()

    try:
        sandbox = _make_sandbox(
            image=metadata["image"], task_dir=os.path.join(os.environ["TMAX_TASK_DATA_DIR"], task_id)
        )
        async with sandbox as box:
            episode.sandbox_start_time = time.monotonic() - t_start
            await _run_episode(episode, policy, box, request_kwargs)
    except SandboxInfraError as e:
        # Only the start can raise here: failures after it are observations inside the loop.
        episode.sandbox_start_time = time.monotonic() - t_start
        await _handle_start_failure(episode, policy, request_kwargs, e)

    metrics = _metrics(episode, time.monotonic() - t_start)
    logger.info(
        f"[{task_id}] episode done: reward={episode.reward} end={episode.end_reason} turns={episode.turns} "
        f"tool_calls={episode.tool_calls} tokens={episode.last_total_tokens} time={metrics['total_time']:.0f}s"
    )
    _write_trace(episode, metadata, metrics)
    return {"reward": episode.reward, "agent_metrics": metrics}


async def abort(args) -> None:
    """Over-sampling abort hook: stop in-flight episodes so the step does not wait on them."""
    path = _abort_generation_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(str(_abort_generation() + 1))
    tmp.replace(path)  # atomic, so a reader never sees a half-written counter
    removed = await (close_live_sandboxes() if backend_name() == "gateway" else remove_owned_sandboxes())
    logger.info(f"TMax abort: signalled in-flight episodes, removed {removed} sandboxes")


def _make_sandbox(image: str, task_dir: str) -> SandboxBase:
    """A container from the configured backend: podman daemons of this job, or a LiteRegistry podman fleet."""
    backend = TmaxGatewaySandbox if backend_name() == "gateway" else TmaxSandbox
    return backend(image=image, task_dir=task_dir)


async def _run_episode(episode: _Episode, policy: AsyncOpenAI, box: TmaxSandbox, request_kwargs: dict) -> None:
    timeout_s = float(os.getenv("TMAX_EPISODE_TIMEOUT_S", "0")) or None
    try:
        await asyncio.wait_for(_loop(episode, policy, box, request_kwargs), timeout=timeout_s)
    except TimeoutError:
        episode.end_reason, episode.reward = "episode_timeout", 0.0
    except SandboxLostError as e:  # the remote container was removed or lost under the episode
        episode.end_reason, episode.sandbox_lost_reason = "sandbox_lost", e.reason
        episode.reward = _lost_sandbox_reward(e.reason)
    except SandboxOOMError:
        episode.end_reason, episode.reward = "oom", 0.0
    except _Aborted:
        episode.end_reason, episode.reward = "aborted", None


async def _handle_start_failure(episode: _Episode, policy: AsyncOpenAI, request_kwargs: dict, error: Exception):
    if os.getenv("TMAX_START_FAILURE", "zero") == "drop":
        logger.warning(f"[{episode.task_id}] sandbox never started, dropping the group: {error}")
        raise error
    # open-instruct SWERL_RESET_FAILURE_ZERO_REWARD=1: the rollout scores 0 and stays in its group. Miles
    # needs a recorded model call to build the sample, so make the smallest one: a single generated token.
    logger.warning(f"[{episode.task_id}] sandbox never started, scoring the episode 0: {error}")
    await policy.chat.completions.create(
        model=os.getenv("AGENT_MODEL_NAME", "model"),
        messages=episode.messages,
        tools=[BASH_TOOL],
        max_tokens=1,
        extra_body={k: v for k, v in request_kwargs.items() if k != "max_tokens"},
    )
    episode.turns = 1
    episode.end_reason, episode.reward = "start_failed", 0.0


async def _loop(episode: _Episode, policy: AsyncOpenAI, box: TmaxSandbox, request_kwargs: dict) -> None:
    max_steps = int(os.getenv("TMAX_MAX_STEPS", "64"))
    model_name = os.getenv("AGENT_MODEL_NAME", "model")
    per_turn_cap = request_kwargs.pop("max_tokens", None)
    messages = episode.messages
    while episode.tool_calls < max_steps:
        max_tokens = _turn_budget(episode, per_turn_cap)
        if max_tokens is not None and max_tokens <= 0:
            episode.end_reason = "context_full"
            return
        t0 = time.monotonic()
        completion = await _race_abort(
            episode,
            policy.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                extra_body=request_kwargs,
            ),
        )
        episode.gen_times.append(time.monotonic() - t0)
        episode.turns += 1
        if completion.usage is not None:
            episode.last_total_tokens = completion.usage.total_tokens
        choice = completion.choices[0]
        # Echo the assistant turn verbatim: the session server prefix-matches the next request against it.
        messages.append(choice.message.model_dump(exclude_none=True))
        if episode.max_seq_len is not None and episode.last_total_tokens > episode.max_seq_len:
            # The last observation overflowed the budget; open-instruct truncates it and stops there.
            episode.end_reason = "context_full"
            return
        if choice.finish_reason == "length":
            episode.end_reason = "length"
            return
        # open-instruct drops calls to tools it does not serve; a turn left without one ends the episode.
        tool_calls = [
            tc for tc in (choice.message.tool_calls or []) if tc.function.name == BASH_TOOL["function"]["name"]
        ]
        if not tool_calls:
            episode.end_reason = "no_tool_call"
            return
        for tool_call in tool_calls:
            if episode.tool_calls >= max_steps:
                break
            episode.tool_calls += 1
            observation, done = await _run_tool_call(episode, box, tool_call)
            if done:
                return
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": observation})


def _turn_budget(episode: _Episode, per_turn_cap: int | None) -> int | None:
    """Per-turn max_tokens: the per-turn cap, shrunk to what is left of max_seq_len."""
    if episode.max_seq_len is None:
        return per_turn_cap
    remaining = episode.max_seq_len - episode.last_total_tokens
    return remaining if per_turn_cap is None else min(per_turn_cap, remaining)


async def _run_tool_call(episode: _Episode, box: TmaxSandbox, tool_call: Any) -> tuple[str, bool]:
    """Run one bash call. Returns (observation, done); a submit runs the tests and sets the reward."""
    try:
        command = json.loads(tool_call.function.arguments or "{}").get("command", "")
    except (json.JSONDecodeError, AttributeError):
        return _FORMAT_ERROR.format(error="Tool call arguments are not a JSON object."), False
    if not command:
        return _FORMAT_ERROR.format(error="'command' parameter is required."), False
    timeout_s = float(os.getenv("TMAX_TOOL_CALL_TIMEOUT_S", "120"))
    t0 = time.monotonic()
    try:
        observation, reward = await _race_abort(episode, asyncio.wait_for(_step(box, command), timeout=timeout_s))
        if reward is None:
            return observation, False
        episode.reward, episode.end_reason = reward, "submitted"
        return observation, True
    except TimeoutError:
        # open-instruct's outer tool_call_timeout (120 s, never overridden in its prod script) wraps the whole
        # env step, including a submit's test run: an overrun is reported to the policy and the episode goes on.
        episode.tool_errors += 1
        return f"Step 'bash' timed out after {timeout_s:g}s. Args: {{'command': {command!r}}}", False
    except (SandboxOOMError, _Aborted):
        raise
    except Exception as e:
        if _abort_generation() != episode.abort_generation:  # the abort hook removed the container
            raise _Aborted() from e
        # open-instruct process_request catches any exception from the env step (its OOM is handled inside the
        # step): it is reported to the policy and the episode goes on. docker-py raises plain ValueErrors too,
        # such as "97 is not a valid stream" when podman's exec framing breaks.
        episode.tool_errors += 1
        return f"Step 'bash' failed: {e}. Args: {{'command': {command!r}}}", False
    finally:
        episode.tool_times.append(time.monotonic() - t0)


async def _step(box: TmaxSandbox, command: str) -> tuple[str, float | None]:
    """One env step: run the command; on the submit marker run the tests. Returns (observation, reward or None)."""
    observation, submitted = await box.bash(command)
    if not submitted:
        return observation, None
    return observation, await box.run_tests()


async def _race_abort(episode: _Episode, coro: Any) -> Any:
    """Await ``coro`` unless the rollout aborts first."""
    task = asyncio.ensure_future(coro)
    abort_wait = asyncio.ensure_future(_wait_for_abort(episode.abort_generation))
    try:
        await asyncio.wait({task, abort_wait}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        abort_wait.cancel()
    if not task.done():
        task.cancel()
        raise _Aborted()
    return task.result()


async def _wait_for_abort(started_under: int) -> None:
    while _abort_generation() == started_under:
        await asyncio.sleep(_ABORT_POLL_S)


def _metrics(episode: _Episode, total_time: float) -> dict[str, Any]:
    return {
        "turns": episode.turns,
        "tool_calls": episode.tool_calls,
        "tool_errors": episode.tool_errors,
        "end_reason": episode.end_reason,
        **({"sandbox_lost_reason": episode.sandbox_lost_reason} if episode.sandbox_lost_reason else {}),
        "submitted": float(episode.end_reason == "submitted"),
        "start_failed": float(episode.end_reason == "start_failed"),
        "sandbox_start_time": episode.sandbox_start_time,
        "model_query_time_sum": sum(episode.gen_times),
        "env_execution_time_sum": sum(episode.tool_times),
        "total_tool_time": sum(episode.tool_times),
        "total_time": total_time,
    }


def _write_trace(episode: _Episode, metadata: dict, metrics: dict) -> None:
    """Append the episode to TMAX_TRACE_DIR/traces-<host>.jsonl; tracing never fails the episode."""
    trace_dir = os.getenv("TMAX_TRACE_DIR")
    if not trace_dir:
        return
    record = {
        "time": time.time(),
        "task_id": episode.task_id,
        "image": metadata.get("image"),
        "reward": episode.reward,
        "metrics": metrics,
        "messages": episode.messages,
    }
    with contextlib.suppress(Exception):
        path = Path(trace_dir) / f"traces-{socket.gethostname()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with rollout_lock(f"tmax-trace-{path.name}"), open(path, "a") as f:
            f.write(line)
