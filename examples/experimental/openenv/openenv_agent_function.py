"""OpenEnv Terminal-Bench-2 <-> miles adapter.

miles selects the policy and calls ``run`` once per episode via
``--custom-agent-function-path openenv_agent_function.run``. The episode drives
the OpenEnv tbench2 env through an agentic loop (reset -> {policy -> exec} ->
evaluate).

The policy is always reached at ``base_url/v1`` through miles' session server,
so token ids + logprobs + loss masks are captured natively (no re-tokenization)
on every turn of the multi-turn episode.

Env vars:
  OPENENV_ENV_URL    base_url of the env server (default: http://localhost:8003).
  OPENENV_MAX_TURNS  multi-turn cap (default: 30)
  OPENENV_MESSAGE_TIMEOUT_S  per-message WS recv timeout (default: 1200). Must
                     exceed the longest single env op the server may
                     legitimately run: in practice the largest
                     [verifier].timeout_sec in your task set (server default
                     900; the official suite declares up to 3600) plus margin
                     -- evaluate times out client-side AFTER the full
                     trajectory was generated, the most expensive point to
                     fail. Also covers cold-image reset (first pull).
  OPENENV_MAX_ROLLOUT_TIME_SECONDS  hard wall-clock cap for one episode (default:
                     3600). An episode that does not return within the limit is
                     terminated and scored reward 0 (bounds long-trajectory
                     stragglers that would otherwise stall the whole rollout batch).
  AGENT_MODEL_NAME   model name sent to the policy (default: "model")
  MILES_ROUTER_EXTERNAL_HOST  optional host rewrite for off-cluster agents

Server contract: the env server must run tbench2_env at or after the
huggingface/OpenEnv#1012 merge (04d259ea6; install per the README) —
canonical tests/test.sh scoring inside the standard ``evaluate`` action, task
WORKDIR resolved server-side, verifier assets withheld. The adapter verifies
the contract on every episode rather than trusting the deployment: an
``evaluate`` reply without the canonical-harness marker is treated as no
verdict and the episode is dropped with a warning (see the guard in
multi_turn).

Per-episode sandbox variants: one sibling module per provider
(``openenv_{daytona,e2b,modal}_agent_function``), each a drop-in
``--custom-agent-function-path`` alternative that runs every episode in its
own cloud sandbox. They reuse this module's agent loop and training wrapper,
supplying only their own run_episode (see the episode-wiring note below);
their env vars are documented there.
"""

import asyncio
import logging
import os
import random
import re
import time
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from miles.rollout.agentic.session import resolve_session_url

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

# The system prompt that teaches a policy this adapter's agent contract, i.e.
# what multi_turn parses: exactly one shell command per turn in a single
# ```bash block, TASK_COMPLETE (no code block) to stop. It lives here, next to
# that parsing logic; make_tbench2_data.py (training prompt data) and
# eval_tbench2_via_api.py (API-policy eval) import it so all consumers stay on
# the one contract.
TB2_AGENT_SYSTEM_PROMPT = (
    "You are an autonomous terminal agent solving a Terminal-Bench task. You will "
    "be given the task instruction, then interact with a real Linux shell. On each "
    "turn respond with EXACTLY ONE shell command inside a single ```bash code block "
    "and nothing else. Inspect the environment, make the required changes, and "
    "verify your work. When you are confident the task is fully complete, reply with "
    "TASK_COMPLETE (with no code block)."
)

# Per-message WS recv timeout. One knob covers three op profiles (reset:
# container create / first image pull; exec: agent commands; evaluate: the
# task's own verifier budget -- task.toml [verifier].timeout_sec, server
# default 900). The default clears the server's default budget with margin;
# raise it (and OPENENV_MAX_ROLLOUT_TIME_SECONDS) for tasks declaring larger
# verifier budgets.
MESSAGE_TIMEOUT_S = float(os.getenv("OPENENV_MESSAGE_TIMEOUT_S", "1200"))

# Hard wall-clock cap for one episode. The per-message timeout above bounds a
# single env op, and OPENENV_MAX_TURNS bounds the turn count, but neither bounds
# total episode time: a long agentic trajectory can loop for turns * (long
# generation) and stall the whole rollout batch (a step finishes only when the
# slowest of all concurrent episodes returns). An episode exceeding this cap is
# terminated (its coroutine cancelled) and scored reward 0.
_MAX_ROLLOUT_TIME_S = float(os.getenv("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "3600"))


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


def _obs_info(result: Any) -> dict:
    """Read the Observation's info dict off a StepResult (empty when absent)."""
    obs = getattr(result, "observation", result)
    return getattr(obs, "info", None) or {}


# Lazy import so the file loads without the env client present at import time.
def load_tbench2() -> dict[str, Any]:
    from tbench2_env import Tbench2Action, Tbench2Env

    return {"env": Tbench2Env, "action": Tbench2Action}


_DEFAULT_ENV_URL = "http://localhost:8003"


# --- Episode wiring -------------------------------------------------------------
# The agent loop (multi_turn) is shared; everything that differs between the
# episode legs enters it as two keyword parameters, filled in only by each
# module's run_episode():
#
#   run_body(env_cls, metadata, body)   how an env comes into being — connect to
#                       the shared server (_shared_run_body below, capacity-
#                       retried) vs create a per-episode sandbox (the sandbox
#                       backend modules).
#   post_episode(env, action_cls)       optional hygiene hook (a long-lived
#                       shared server accumulates trial dirs; a per-episode
#                       sandbox lives only for its episode and needs nothing).
#
# Every agent-function module exposes the same two entries: run() for miles
# (session-server policy wiring + training failure semantics) and
# run_episode() for callers that bring their own policy client and own
# timeout/failure semantics (eval_tbench2_via_api).


async def _shared_run_body(env_cls: Any, metadata: dict[str, Any], body: Callable[[Any], Any]) -> Any:
    """Run *body* against the one shared env server at OPENENV_ENV_URL."""
    env_url = os.getenv("OPENENV_ENV_URL", _DEFAULT_ENV_URL)
    return await _with_env(env_cls, env_url, body)


async def _purge_trial_dirs(env: Any, action_cls: Any) -> None:
    # Shared-server disk hygiene (a Daytona sandbox needs none of
    # this — it is deleted when its episode ends): the tbench2 env server
    # (TB2_OUTPUT_DIR=/tmp/tbench2_env_runs) leaves a per-episode trial dir
    # under that path after every episode, which fills the sandbox overlay
    # disk and trips ENOSPC. One episode holds the sandbox at a time, so it
    # is safe to purge them here.
    #
    # BUT the same dir also holds repo_cache/ (TB2_CACHE_DIR defaults to
    # output_dir/repo_cache) -- the shared terminal-bench-2 checkout that
    # reset() clones once and every later episode reads its task from
    # (repo_cache/terminal-bench-2-main/<task>). A blanket `rm -rf .../*`
    # wiped repo_cache too, so on a pooled/reused sandbox every episode
    # after the first either re-cloned the whole repo (huge) or raced into
    # "Task path not found", collapsing effective concurrency and exploding
    # step time. Preserve repo_cache; delete only the ephemeral per-trial
    # dirs beside it.
    try:
        await env.step(
            action_cls(
                action_type="exec",
                command=(
                    "find /tmp/tbench2_env_runs -mindepth 1 -maxdepth 1 "
                    "! -name repo_cache -exec rm -rf {} + 2>/dev/null || true"
                ),
            )
        )
    except Exception:
        pass


async def _with_env(env_cls: Any, env_url: str, body: Callable[[Any], Any]) -> Any:
    """Open an env session and run ``body(env)``, retrying while a slot is busy."""
    deadline = asyncio.get_event_loop().time() + _CAPACITY_MAX_WAIT_S
    while True:
        try:
            async with env_cls(base_url=env_url, message_timeout_s=MESSAGE_TIMEOUT_S) as env:
                return await body(env)
        except Exception as e:
            if _is_retryable_env_error(e) and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(random.uniform(*_CAPACITY_BACKOFF_S))
                continue
            raise


async def multi_turn(
    classes: dict[str, Any],
    policy: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
    *,
    run_body: Callable[..., Any],
    post_episode: Callable[..., Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Agentic loop: reset(task) -> {policy -> exec -> feed output back} -> evaluate (tbench2).

    The policy emits one shell command per turn (a ```bash block or the bare
    reply), executed by the server in the task image's real WORKDIR; the loop
    ends when the policy stops emitting a command, says TASK_COMPLETE, or hits
    OPENENV_MAX_TURNS. Scoring is the standard ``evaluate`` action: the server
    runs the task's tests/test.sh and reports the verdict (see the module
    docstring for the required server contract). The returned agent metrics
    label the stop as ``task_complete``, ``no_command``, ``max_turns``, or
    ``length``.
    """
    action_cls = classes["action"]
    task_id = metadata.get("task_id") or metadata.get("task_name")
    max_turns = int(os.getenv("OPENENV_MAX_TURNS", "30"))

    async def body(env: Any) -> tuple[float | None, int, str, list[float], list[float], float, float]:
        # Per-turn wall-clock timings. gen_times[i] is turn i's policy generation
        # latency; tool_times[i] is turn i's env.step(exec) latency. reset_time and
        # eval_time bracket the one-off reset() and the final evaluate() env steps.
        gen_times: list[float] = []
        tool_times: list[float] = []

        t0 = time.monotonic()
        reset_result = await (env.reset(task_id=task_id) if task_id else env.reset())
        reset_time = time.monotonic() - t0
        instruction = _obs_field(reset_result, "instruction")
        convo = list(messages)
        if instruction:
            convo.append({"role": "user", "content": instruction})

        turns = 0
        end_reason = "max_turns"
        while turns < max_turns:
            turns += 1
            t0 = time.monotonic()
            completion = await policy.chat.completions.create(
                model=model_name, messages=convo, extra_body=request_kwargs
            )
            gen_times.append(time.monotonic() - t0)
            choice = completion.choices[0]
            message = choice.message
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

            if choice.finish_reason == "length":
                end_reason = "length"
                break

            command = _strip_fence(reply) if "```" in reply else reply.strip()
            if not command:
                end_reason = "no_command"
                break
            if command.upper().startswith("TASK_COMPLETE"):
                end_reason = "task_complete"
                break

            t0 = time.monotonic()
            step_result = await env.step(action_cls(action_type="exec", command=command))
            tool_times.append(time.monotonic() - t0)
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

        t0 = time.monotonic()
        eval_result = await env.step(action_cls(action_type="evaluate"))
        eval_time = time.monotonic() - t0
        # No canonical verdict -> reward None (the training wrapper drops the
        # sample instead of ingesting a false-negative 0):
        #   - reward=None / `error` set: the scoring step itself errored
        #     server-side (toolkit timeout, staging I/O) -- not tests failing.
        #   - harness marker absent: the server scored, but not through the
        #     canonical tests/test.sh (a tbench2_env install predating the
        #     contract in the module docstring, or a task dir without test.sh
        #     scored by the server's pytest fallback). The reward LOOKS
        #     valid, which is exactly why it must not be trusted: source
        #     preflight is impossible against a remote server, so this marker
        #     is the contract check.
        raw_reward = getattr(eval_result, "reward", None)
        eval_error = _obs_field(eval_result, "error")
        harness = str(_obs_info(eval_result).get("harness", ""))
        if raw_reward is None or eval_error or harness != "tests/test.sh":
            logger.warning(
                "OpenEnv tbench2 evaluate produced no canonical verdict "
                f"(error={eval_error!r}, harness={harness!r}); dropping episode"
            )
            reward = None
        else:
            reward = float(raw_reward)

        if post_episode is not None:
            await post_episode(env, action_cls)

        return reward, turns, end_reason, gen_times, tool_times, reset_time, eval_time

    result = await run_body(classes["env"], metadata, body)
    reward, turns, end_reason, gen_times, tool_times, reset_time, eval_time = result
    total_gen_time = sum(gen_times)
    # non_generation_time = everything the rollout spent outside policy generation:
    # per-turn exec latency plus the one-off reset() and evaluate() env steps. Feeds
    # Sample.non_generation_time so miles' throughput accounting subtracts env time.
    total_tool_time = sum(tool_times) + reset_time + eval_time
    return reward, {
        "turns": turns,
        "end_reason": end_reason,
        "tool_calls": len(tool_times),
        "gen_times": gen_times,
        "tool_times": tool_times,
        "reset_time": reset_time,
        "eval_time": eval_time,
        "total_gen_time": total_gen_time,
        "total_tool_time": total_tool_time,
    }


async def run_episode(
    policy: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """One episode against the shared env server, with the caller's own policy.

    The direct-drive entry (see the module docstring): returns the loop's raw
    ``(reward, agent_metrics)``; wall-clock caps and failure semantics are the
    caller's. miles goes through run() instead.
    """
    return await multi_turn(
        load_tbench2(),
        policy,
        model_name,
        messages,
        request_kwargs,
        metadata,
        run_body=_shared_run_body,
        post_episode=_purge_trial_dirs,
    )


async def run_for_training(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    run_episode_fn: Callable[..., Any],
) -> dict[str, Any] | None:
    """miles-side wrapper around one episode: session-server policy wiring plus
    training failure semantics (timeout -> reward 0, no verdict -> drop sample).

    Shared by every agent-function module: each passes its own run_episode.
    """
    request_kwargs = request_kwargs or {}
    metadata = metadata or {}

    session_url = resolve_session_url(base_url)
    model_name = os.getenv("AGENT_MODEL_NAME", os.getenv("SWE_AGENT_MODEL_NAME", "model"))

    policy = AsyncOpenAI(base_url=session_url, api_key="EMPTY")
    messages = _extract_messages(prompt)

    try:
        # Hard wall-clock cap: cancel the episode if it overruns and score it 0.
        # wait_for cancels the coroutine, so any in-flight policy call / env.step
        # is interrupted and the env session is closed by the env context manager
        # during cancellation cleanup.
        reward, agent_metrics = await asyncio.wait_for(
            run_episode_fn(policy, model_name, messages, request_kwargs, metadata),
            timeout=_MAX_ROLLOUT_TIME_S,
        )
    except asyncio.TimeoutError:
        logger.warning(f"OpenEnv tbench2 episode exceeded {_MAX_ROLLOUT_TIME_S:.0f}s; " "terminating with reward 0")
        # eval_report empty: the episode was cancelled before evaluate ever
        # ran, so there is no pytest report to surface.
        return {
            "reward": 0.0,
            "exit_status": "timeout",
            "eval_report": {},
            "agent_metrics": {"timed_out": 1},
        }
    except Exception as e:
        logger.error(f"OpenEnv tbench2 episode failed: {e}", exc_info=True)
        return None
    finally:
        await policy.close()

    # No canonical verdict (infra/harness failure or a non-canonical server,
    # not a legitimate task failure -- see the guard in multi_turn). Drop the
    # sample: returning it as reward 0.0 would inject a false negative into
    # training.
    if reward is None:
        logger.warning("OpenEnv tbench2 episode produced no canonical reward; dropping sample")
        return None

    # eval_report is intentionally empty: the server's evaluate reports only
    # the scalar reward (plus the harness marker). The detailed pytest CTRF
    # report is written inside the sandbox at /logs/verifier/ctrf.json and is
    # deliberately not captured back to the trainer, which consumes only
    # `reward`.
    return {
        "reward": reward,
        "exit_status": "completed",
        "eval_report": {},
        "agent_metrics": agent_metrics,
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one OpenEnv tbench2 episode via the trained policy (shared env server)."""
    return await run_for_training(base_url, prompt, request_kwargs, metadata, run_episode)
