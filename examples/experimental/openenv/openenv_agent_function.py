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
  OPENENV_MESSAGE_TIMEOUT_S  per-message WS recv timeout (default: 600; docker-mode
                     reset/exec/pytest routinely exceed the client default of 60)
  OPENENV_MAX_ROLLOUT_TIME_SECONDS  hard wall-clock cap for one episode (default:
                     3600). An episode that does not return within the limit is
                     terminated and scored reward 0 (bounds long-trajectory
                     stragglers that would otherwise stall the whole rollout batch).
  AGENT_MODEL_NAME   model name sent to the policy (default: "model")
  MILES_ROUTER_EXTERNAL_HOST  optional host rewrite for off-cluster agents
  OPENENV_TASK_WORKDIR  container dir every agent command + eval runs in (default:
                     /app, the TB2 task image WORKDIR). Empty string disables the
                     prefix. Needed because upstream OpenEnv defaults to /task.
  OPENENV_TB2_TESTS_SRC  where the upstream env stages the task's tests inside the
                     container (default: /task/tests); copied to /tests for test.sh.

Per-task Daytona sandbox backend (alternative to OPENENV_ENV_URL): every episode
gets its OWN cloud sandbox built from the task's OFFICIAL image plus an env
server layer, deleted when the episode ends. The sandbox recipe lives in
``tbench2_env.task_snapshots`` -- ``tbench2_env`` is OpenEnv's Terminal-Bench-2
environment package (``envs/tbench2_env`` in huggingface/openenv), and this
backend needs the patched branch proposed in openenv#965 + #966. Full per-task
image fidelity with zero shared infrastructure (no Docker host, no resident
env server) and zero cross-episode state leakage.
  OPENENV_TB2_TASKS_DIR        path to a terminal-bench-2 checkout: build the
                     sandbox declaratively per episode. Daytona caches image
                     layers by definition hash, so only the first episode of a
                     task builds (~10 min); repeats start in ~1 min. No named
                     snapshots, so no org snapshot quota.
  DAYTONA_API_KEY              required in per-task mode.
  OPENENV_DAYTONA_CREATE_CONCURRENCY  max in-flight sandbox creates (default 4).
  OPENENV_DAYTONA_READY_TIMEOUT_S     server-ready wait per sandbox (default 300).
  TB2_COMMAND_TIMEOUT_S        per-exec timeout inside the sandbox (default 900).
"""

import asyncio
import logging
import os
import random
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
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

# The system prompt that teaches a policy this adapter's agent contract, i.e.
# what _multi_turn parses: exactly one shell command per turn in a single
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

# --- Adapter-driven Terminal-Bench-2 fidelity --------------------------------
# Upstream OpenEnv's Tbench2DockerEnvironment runs the task container with workdir
# /task (a copy of the task *source*) and scores via bare `pytest tests/` there.
# Real TB2 tasks live at /app (the task image's WORKDIR) and are scored by the
# task's canonical tests/test.sh, which pins the pytest toolchain, copies test.py
# into /app, runs test_outputs.py, and writes the binary result to
# /logs/verifier/reward.txt. We reproduce that faithfully from the adapter --
# without patching OpenEnv or vendoring it -- by (a) running every agent command
# in _TASK_WORKDIR and (b) driving the canonical harness through a plain `exec`
# step instead of the env's built-in (non-canonical) `evaluate` action.
#
# This assumes the env server is UNMODIFIED upstream, which copies the task dir
# (tests included) into the container at _TB2_TESTS_SRC.
_TASK_WORKDIR = os.getenv("OPENENV_TASK_WORKDIR", "/app")
_TB2_TESTS_SRC = os.getenv("OPENENV_TB2_TESTS_SRC", "/task/tests")

# The eval exec echoes reward.txt on this marker so we can parse it out of stdout.
_REWARD_MARKER = "__TB2_REWARD__:"
# test.sh's exit code, echoed on its own marker purely for diagnostics: when a
# sample is dropped for having no recoverable reward, a nonzero rc points at a
# test.sh crash (infra/harness failure) vs. a clean run that wrote no verdict.
# It does NOT drive the drop decision -- a nonzero rc from merely-failing tests
# is a legitimate reward 0, not an infra error.
_TESTSH_RC_MARKER = "__TB2_TESTSH_RC__:"
# Honor an empty _TASK_WORKDIR (workdir prefix disabled) the same way
# _apply_workdir does, instead of silently forcing /app.
_EVAL_CD_CMD = f"cd {_TASK_WORKDIR} && " if _TASK_WORKDIR else ""
_CANONICAL_EVAL_CMD = (
    # rm the reward file first so a stale one can never be read back if test.sh
    # fails to run (e.g. in a reused sandbox where /logs survives across episodes).
    "mkdir -p /tests /logs/verifier && rm -f /logs/verifier/reward.txt && "
    f"cp -a {_TB2_TESTS_SRC}/. /tests/ 2>/dev/null || true; "
    f"{_EVAL_CD_CMD}bash /tests/test.sh > /tmp/tb2_testsh.log 2>&1; "
    # $? here is test.sh's exit code (captured before any other command runs).
    f"echo {_TESTSH_RC_MARKER}$?; "
    f"echo {_REWARD_MARKER}$(cat /logs/verifier/reward.txt 2>/dev/null)"
)


def _apply_workdir(command: str) -> str:
    """Prefix an agent command so it runs in the real task workdir (/app)."""
    if not _TASK_WORKDIR:
        return command
    return f"cd {_TASK_WORKDIR} && {command}"


def _parse_reward_marker(output: str) -> float | None:
    """Parse the reward.txt value the canonical-eval exec echoed on its marker line.

    Returns None when no reward can be recovered -- no marker line, an empty
    value (reward.txt absent, i.e. test.sh never wrote a verdict), or a
    non-numeric value. These are infra/harness failures, not a task the agent
    legitimately failed, so the caller drops the sample rather than scoring a
    false 0.0 that would pollute the training signal. A genuine failure writes
    reward.txt = 0 and is returned as 0.0.
    """
    for line in output.splitlines()[::-1]:
        if _REWARD_MARKER in line:
            raw = line.split(_REWARD_MARKER, 1)[1].strip()
            if not raw:
                return None
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def _parse_testsh_rc(output: str) -> int | None:
    """Parse test.sh's exit code off its marker line (diagnostic only, may be absent)."""
    for line in output.splitlines()[::-1]:
        if _TESTSH_RC_MARKER in line:
            raw = line.split(_TESTSH_RC_MARKER, 1)[1].strip()
            try:
                return int(raw)
            except ValueError:
                return None
    return None


# Per-message WS recv timeout. Docker-mode tbench2 reset (container create),
# exec, and evaluate (pytest) each routinely exceed the EnvClient default of 60s.
_MESSAGE_TIMEOUT_S = float(os.getenv("OPENENV_MESSAGE_TIMEOUT_S", "600"))

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


# Lazy import so the file loads without the env client present at import time.
def _load_tbench2() -> dict[str, Any]:
    from tbench2_env import Tbench2Action, Tbench2Env

    return {"env": Tbench2Env, "action": Tbench2Action}


_DEFAULT_ENV_URL = "http://localhost:8003"


# --- Per-task Daytona sandboxes (one per episode) -----------------------------
# The per-task image recipe (official task image + env server layer) lives in
# OpenEnv's tbench2_env package (tbench2_env.task_snapshots). Each episode
# materializes it declaratively from the Image definition, read off the local
# TB2 checkout (OPENENV_TB2_TASKS_DIR); repeat creates hit Daytona's build
# cache, and no named snapshot is involved.
#
# The sandbox's env server is the PATCHED tbench2_env baked by task_snapshots
# (canonical tests/test.sh scoring built into `evaluate`, per-task WORKDIR
# resolved server-side, configurable exec timeout). The adapter-driven fidelity
# machinery above (_apply_workdir / _CANONICAL_EVAL_CMD) exists to compensate
# for an UNMODIFIED upstream server and is deliberately not applied on this
# backend -- each leg matches what its server actually provides.
#
# Daytona rate-limits sandbox creation (ThrottlerException: Too Many Requests).
# A rollout fans out many episodes at once; cap in-flight creates process-wide
# and retry throttled ones with jittered exponential backoff.
_CREATE_CONCURRENCY = int(os.getenv("OPENENV_DAYTONA_CREATE_CONCURRENCY", "4"))
_CREATE_MAX_RETRIES = int(os.getenv("OPENENV_DAYTONA_CREATE_MAX_RETRIES", "8"))
_CREATE_BACKOFF_BASE_S = float(os.getenv("OPENENV_DAYTONA_CREATE_BACKOFF_BASE_S", "2.0"))
_CREATE_BACKOFF_CAP_S = float(os.getenv("OPENENV_DAYTONA_CREATE_BACKOFF_CAP_S", "30.0"))
_READY_TIMEOUT_S = float(os.getenv("OPENENV_DAYTONA_READY_TIMEOUT_S", "300"))
_COMMAND_TIMEOUT_S = int(os.getenv("TB2_COMMAND_TIMEOUT_S", "900"))

_create_sem: asyncio.Semaphore | None = None


def _per_task_mode() -> bool:
    """True when episodes run in per-task Daytona sandboxes instead of OPENENV_ENV_URL."""
    return bool(os.getenv("OPENENV_TB2_TASKS_DIR", "").strip())


def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because Daytona rate-limited it.

    The daytona SDK is a lazy, per-task-mode-only dependency (shared-server
    users don't install it), so its exception classes cannot be imported at
    module scope -- but by the time a create has FAILED, daytona has
    necessarily been imported, so the typed check happens here. The SDK
    normalizes HTTP 429 to DaytonaRateLimitError; keep the text match as a
    fallback for older SDKs and server messages that only surface as text
    (e.g. "ThrottlerException: Too Many Requests").
    """
    try:
        from daytona.common.errors import DaytonaRateLimitError

        if isinstance(exc, DaytonaRateLimitError):
            return True
    except ImportError:  # pragma: no cover - only without the daytona SDK
        pass
    s = str(exc).lower()
    return "throttler" in s or "too many requests" in s or "429" in s


def _get_create_sem() -> asyncio.Semaphore:
    global _create_sem
    if _create_sem is None:
        _create_sem = asyncio.Semaphore(_CREATE_CONCURRENCY)
    return _create_sem


def _start_declarative(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    from tbench2_env import task_snapshots

    daytona = task_snapshots._make_daytona()
    sandbox, url = task_snapshots.create_task_sandbox(
        daytona,
        Path(tasks_dir) / task_id,
        command_timeout_s=_COMMAND_TIMEOUT_S,
        ready_timeout_s=_READY_TIMEOUT_S,
    )
    return (lambda: daytona.delete(sandbox)), url


async def _start_task_sandbox(task_id: str) -> tuple[Any, str]:
    """Create one per-task sandbox with the env server running.

    Returns (close_fn, base_url); close_fn deletes the sandbox. Creation is
    throttled process-wide and retried on Daytona rate limits.
    """
    tasks_dir = os.getenv("OPENENV_TB2_TASKS_DIR", "").strip()

    def _start() -> tuple[Any, str]:
        return _start_declarative(task_id, tasks_dir)

    attempt = 0
    while True:
        try:
            # Hold the semaphore only for the create attempt; release it during
            # backoff so other episodes keep the pipeline full.
            async with _get_create_sem():
                return await asyncio.to_thread(_start)
        except Exception as e:
            if not _is_throttle_error(e) or attempt >= _CREATE_MAX_RETRIES:
                raise
            attempt += 1
            delay = min(
                _CREATE_BACKOFF_CAP_S,
                _CREATE_BACKOFF_BASE_S * (2 ** (attempt - 1)),
            ) * (0.5 + random.random())
            logger.warning(
                f"Daytona create throttled for {task_id} "
                f"(attempt {attempt}/{_CREATE_MAX_RETRIES}); retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)


@asynccontextmanager
async def _episode_env(env_cls: Any, metadata: dict[str, Any]):
    """Yield a connected env client on a fresh per-task sandbox; delete it after.

    Per-task Daytona mode only (_per_task_mode() is True). The shared
    OPENENV_ENV_URL backend goes through _with_env instead.
    """
    task_id = metadata.get("task_id") or metadata.get("task_name")
    if not task_id:
        raise ValueError("per-task sandbox mode requires metadata['task_id']")
    close_fn, url = await _start_task_sandbox(str(task_id))
    try:
        async with env_cls(base_url=url, message_timeout_s=_MESSAGE_TIMEOUT_S) as env:
            yield env
    finally:
        try:
            await asyncio.to_thread(close_fn)
        except Exception as e:
            logger.warning(f"Failed to delete sandbox for {task_id}: {e}")


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


async def _multi_turn(
    classes: dict[str, Any],
    policy: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Agentic loop: reset(task) -> {policy -> exec -> feed output back} -> evaluate (tbench2).

    The policy emits one shell command per turn (a ```bash block or the bare
    reply), executed in the real task workdir; the loop ends when the policy
    stops emitting a command, says TASK_COMPLETE, or hits OPENENV_MAX_TURNS.

    Scoring depends on the backend, matching what its env server provides:
    shared-server episodes run the task's canonical tests/test.sh via an
    ``exec`` step and parse /logs/verifier/reward.txt (faithful to
    Terminal-Bench-2 against an unmodified upstream server); per-task-sandbox
    episodes use the standard ``evaluate`` action, because the sandbox's
    patched server runs that same canonical test.sh natively (and resolves the
    task WORKDIR itself, so no _apply_workdir prefix either).
    """
    action_cls = classes["action"]
    task_id = metadata.get("task_id") or metadata.get("task_name")
    max_turns = int(os.getenv("OPENENV_MAX_TURNS", "30"))
    per_task = _per_task_mode()

    async def body(env: Any) -> tuple[float | None, int, list[float], list[float], float, float, int | None]:
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
        while turns < max_turns:
            turns += 1
            t0 = time.monotonic()
            completion = await policy.chat.completions.create(
                model=model_name, messages=convo, extra_body=request_kwargs
            )
            gen_times.append(time.monotonic() - t0)
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

            t0 = time.monotonic()
            # Per-task sandboxes run the server-side toolkit in the task's real
            # WORKDIR already; only the shared upstream server needs the prefix.
            exec_command = command if per_task else _apply_workdir(command)
            step_result = await env.step(action_cls(action_type="exec", command=exec_command))
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
        if per_task:
            # Per-task sandbox: the patched server scores via the standard
            # evaluate action (runs the canonical test.sh natively), so there
            # is no adapter-side test.sh exit-code marker to parse.
            eval_result = await env.step(action_cls(action_type="evaluate"))
            eval_time = time.monotonic() - t0
            reward = float(getattr(eval_result, "reward", 0.0) or 0.0)
            testsh_rc = None
        else:
            # Shared upstream server: adapter-driven canonical exec + marker parse.
            eval_result = await env.step(action_cls(action_type="exec", command=_CANONICAL_EVAL_CMD))
            eval_time = time.monotonic() - t0
            eval_output = _obs_field(eval_result, "output")
            reward = _parse_reward_marker(eval_output)
            testsh_rc = _parse_testsh_rc(eval_output)

        # rm-hack (shared-server backend only -- a per-task sandbox is deleted
        # when its episode ends, so nothing accumulates there):
        # the tbench2 env server (TB2_OUTPUT_DIR=/tmp/tbench2_env_runs)
        # leaves a per-episode trial dir under that path after every episode, which
        # fills the sandbox overlay disk and trips ENOSPC. One episode holds the
        # sandbox at a time, so it is safe to purge them here.
        #
        # BUT the same dir also holds repo_cache/ (TB2_CACHE_DIR defaults to
        # output_dir/repo_cache) -- the shared terminal-bench-2 checkout that
        # reset() clones once and every later episode reads its task from
        # (repo_cache/terminal-bench-2-main/<task>). A blanket `rm -rf .../*` wiped
        # repo_cache too, so on a pooled/reused sandbox every episode after the first
        # either re-cloned the whole repo (huge) or raced into "Task path not found",
        # collapsing effective concurrency and exploding step time. Preserve
        # repo_cache; delete only the ephemeral per-trial dirs beside it.
        if not per_task:
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

        return reward, turns, gen_times, tool_times, reset_time, eval_time, testsh_rc

    if per_task:
        async with _episode_env(classes["env"], metadata) as env:
            result = await body(env)
    else:
        env_url = os.getenv("OPENENV_ENV_URL", _DEFAULT_ENV_URL)
        result = await _with_env(classes["env"], env_url, body)
    reward, turns, gen_times, tool_times, reset_time, eval_time, testsh_rc = result
    total_gen_time = sum(gen_times)
    # non_generation_time = everything the rollout spent outside policy generation:
    # per-turn exec latency plus the one-off reset() and evaluate() env steps. Feeds
    # Sample.non_generation_time so miles' throughput accounting subtracts env time.
    total_tool_time = sum(tool_times) + reset_time + eval_time
    return reward, {
        "turns": turns,
        "tool_calls": len(tool_times),
        "gen_times": gen_times,
        "tool_times": tool_times,
        "reset_time": reset_time,
        "eval_time": eval_time,
        "total_gen_time": total_gen_time,
        "total_tool_time": total_tool_time,
        "testsh_rc": testsh_rc,
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one OpenEnv tbench2 episode via the trained policy."""
    request_kwargs = request_kwargs or {}
    metadata = metadata or {}

    classes = _load_tbench2()
    session_url = _resolve_session_url(base_url)
    model_name = os.getenv("AGENT_MODEL_NAME", os.getenv("SWE_AGENT_MODEL_NAME", "model"))

    policy = AsyncOpenAI(base_url=session_url, api_key="EMPTY")
    messages = _extract_messages(prompt)

    try:
        # Hard wall-clock cap: cancel the episode if it overruns and score it 0.
        # wait_for cancels the coroutine, so any in-flight policy call / env.step
        # is interrupted and the env session is closed by the env context manager
        # during cancellation cleanup.
        reward, agent_metrics = await asyncio.wait_for(
            _multi_turn(classes, policy, model_name, messages, request_kwargs, metadata),
            timeout=_MAX_ROLLOUT_TIME_S,
        )
    except asyncio.TimeoutError:
        logger.warning(f"OpenEnv tbench2 episode exceeded {_MAX_ROLLOUT_TIME_S:.0f}s; " "terminating with reward 0")
        # eval_report empty: the episode was cancelled before the canonical
        # eval ever ran, so there is no pytest report to surface.
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

    # No recoverable reward means the canonical harness never produced a verdict
    # (infra/harness failure, not a legitimate task failure). Drop the sample --
    # returning it as reward 0.0 would inject a false negative into training.
    if reward is None:
        logger.warning(
            "OpenEnv tbench2 episode produced no canonical reward "
            f"(test.sh exit code={agent_metrics.get('testsh_rc')}); "
            "infra/harness failure, dropping sample"
        )
        return None

    # eval_report is intentionally empty: the canonical-eval marker protocol
    # (see _REWARD_MARKER) echoes back only the scalar reward. The detailed
    # pytest CTRF report is written inside the sandbox at
    # /logs/verifier/ctrf.json and is deliberately not captured back to the
    # trainer, which consumes only `reward`.
    return {
        "reward": reward,
        "exit_status": "completed",
        "eval_report": {},
        "agent_metrics": agent_metrics,
    }
