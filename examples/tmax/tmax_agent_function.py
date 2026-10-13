"""TMax Terminal-RL agent loop for ``agentic_tool_call.generate``.

Miles calls ``run`` once per episode via ``--custom-agent-function-path
tmax_agent_function.run``. The policy is reached through the session server at
``base_url/v1``, so token ids, logprobs and loss masks are recorded server-side
on every turn; this module only owns the loop and the sandbox.

The loop mirrors open-instruct's Terminal-RL rollout (``vllm_utils.process_request``
with the ``swerl_vanillux_sandbox`` env and ``--tool_call_format_error_feedback``
off):

  start a fresh container for the task
  repeat up to TMAX_MAX_STEPS tool calls:
      chat completion with the single `bash` tool
      no tool call / length-truncated turn / context full  -> stop, reward 0
      run each bash call; the submit marker runs the tests  -> reward, stop

Failure semantics (see docs/deep-dive/agentic-rollout-internals.md section 4.5):

* sandbox cannot be started: raised before the first model call, so the
  session has no records, the sample is ABORTED and the group is dropped;
* sandbox infra failure after the first model call: ``reward=None``, which the
  missing-reward filter turns into a dropped group (open-instruct instead
  scores these 0 under SWERL_RESET_FAILURE_ZERO_REWARD=1);
* sandbox OOM or episode wall-clock cap: reward 0, as in open-instruct;
* over-sampling abort: ``reward=None``; the group is discarded anyway.

Each call runs in its own process under ``--custom-agent-function-mode subproc``
(the default), so the abort hook, which runs in the rollout process, reaches
in-flight episodes through a generation counter in a file scoped to that rollout
process, plus removal of the rollout's containers to cut long commands short.

Env vars:
  TMAX_TASK_DATA_DIR       directory holding one sub-directory per task_id (required).
  TMAX_MAX_STEPS           max bash calls per episode (default: 64).
  TMAX_EPISODE_TIMEOUT_S   wall-clock cap per episode (default: 3600).
  AGENT_MODEL_NAME         model name sent to the policy (default: "model").
  plus the sandbox knobs documented in tmax_sandbox.py.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tmax_sandbox import SandboxInfraError, SandboxOOMError, TmaxSandbox, remove_owned_sandboxes

from miles.rollout.agentic.rollout_limits import current_scope
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


class _Aborted(Exception):
    """The rollout step aborted this episode (over-sampling); its group is discarded."""


_ABORT_POLL_S = 1.0


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
    end_reason: str = "max_steps"
    reward: float | None = 0.0
    last_total_tokens: int = 0
    gen_times: list[float] = field(default_factory=list)
    tool_times: list[float] = field(default_factory=list)


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
        task_id=task_id, max_seq_len=metadata.get("max_seq_len"), abort_generation=_abort_generation()
    )
    policy = AsyncOpenAI(base_url=openai_session_url(base_url), api_key="EMPTY", timeout=3600)
    task_dir = os.path.join(os.environ["TMAX_TASK_DATA_DIR"], task_id)
    t_start = time.monotonic()

    # Started outside the try: a sandbox that never comes up leaves no model call recorded,
    # so the sample is ABORTED and the group dropped instead of training on a fake 0.
    async with TmaxSandbox(image=metadata["image"], task_dir=task_dir) as box:
        try:
            await asyncio.wait_for(
                _loop(episode, policy, box, list(prompt), dict(request_kwargs or {})),
                timeout=float(os.getenv("TMAX_EPISODE_TIMEOUT_S", "3600")),
            )
        except TimeoutError:
            episode.end_reason, episode.reward = "episode_timeout", 0.0
        except SandboxOOMError:
            episode.end_reason, episode.reward = "oom", 0.0
        except _Aborted:
            episode.end_reason, episode.reward = "aborted", None
        except SandboxInfraError as e:
            if _abort_generation() != episode.abort_generation:  # the abort hook removed the container
                episode.end_reason, episode.reward = "aborted", None
            else:
                logger.warning(f"[{task_id}] sandbox infra failure, dropping group: {e}")
                episode.end_reason, episode.reward = "infra_error", None

    metrics = _metrics(episode, time.monotonic() - t_start)
    logger.info(
        f"[{task_id}] episode done: reward={episode.reward} end={episode.end_reason} turns={episode.turns} "
        f"tool_calls={episode.tool_calls} tokens={episode.last_total_tokens} time={metrics['total_time']:.0f}s"
    )
    return {"reward": episode.reward, "agent_metrics": metrics}


async def abort(args) -> None:
    """Over-sampling abort hook: stop in-flight episodes so the step does not wait on them."""
    path = _abort_generation_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(str(_abort_generation() + 1))
    tmp.replace(path)  # atomic, so a reader never sees a half-written counter
    removed = await remove_owned_sandboxes()
    logger.info(f"TMax abort: signalled in-flight episodes, removed {removed} sandboxes")


async def _loop(
    episode: _Episode, policy: AsyncOpenAI, box: TmaxSandbox, messages: list[dict], request_kwargs: dict
) -> None:
    max_steps = int(os.getenv("TMAX_MAX_STEPS", "64"))
    model_name = os.getenv("AGENT_MODEL_NAME", "model")
    per_turn_cap = request_kwargs.pop("max_tokens", None)
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
        if choice.finish_reason == "length":
            episode.end_reason = "length"
            return
        if not choice.message.tool_calls:
            episode.end_reason = "no_tool_call"
            return
        for tool_call in choice.message.tool_calls:
            episode.tool_calls += 1
            observation, submitted = await _run_tool_call(episode, box, tool_call)
            if submitted:
                episode.reward = await _race_abort(episode, box.run_tests())
                episode.end_reason = "submitted"
                return
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": observation})


def _turn_budget(episode: _Episode, per_turn_cap: int | None) -> int | None:
    """Per-turn max_tokens: the per-turn cap, shrunk to what is left of max_seq_len."""
    if episode.max_seq_len is None:
        return per_turn_cap
    remaining = episode.max_seq_len - episode.last_total_tokens
    return remaining if per_turn_cap is None else min(per_turn_cap, remaining)


async def _run_tool_call(episode: _Episode, box: TmaxSandbox, tool_call: Any) -> tuple[str, bool]:
    if tool_call.function.name != "bash":
        return _FORMAT_ERROR.format(error=f"Unknown tool '{tool_call.function.name}'. The only tool is `bash`."), False
    try:
        command = json.loads(tool_call.function.arguments or "{}").get("command", "")
    except (json.JSONDecodeError, AttributeError):
        return _FORMAT_ERROR.format(error="Tool call arguments are not a JSON object."), False
    if not command:
        return _FORMAT_ERROR.format(error="'command' parameter is required."), False
    t0 = time.monotonic()
    try:
        return await _race_abort(episode, box.bash(command))
    finally:
        episode.tool_times.append(time.monotonic() - t0)


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
        "end_reason": episode.end_reason,
        "submitted": float(episode.end_reason == "submitted"),
        "model_query_time_sum": sum(episode.gen_times),
        "env_execution_time_sum": sum(episode.tool_times),
        "total_tool_time": sum(episode.tool_times),
        "total_time": total_time,
    }
