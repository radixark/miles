"""OpenEnv SWE-bench-style <-> miles adapter.

A sibling of ``examples/experimental/openenv`` (Terminal-Bench-2). Same env
server, same agentic loop, same reward-marker protocol -- only the task family
differs. SWE-bench-style tasks are the SWE-Rebench-V2 "donor" variants: a git
repository baked into a prebuilt image, an instruction describing a code change,
and a shell verifier (``tests/test.sh`` + ``tests/config.json``) that runs the
task's ``test_command`` and prints ``RESULT: PASSED`` / ``RESULT: FAILED``.

miles selects the policy and calls ``run`` once per episode via
``--custom-agent-function-path swebench_agent_function.run``. The episode drives
the OpenEnv env through an agentic loop (reset -> {policy -> exec} -> evaluate).

The policy is always reached at ``base_url/v1`` through miles' session server,
so token ids + logprobs + loss masks are captured natively (no re-tokenization)
on every turn of the multi-turn episode.

Two things make this family distinct from TB2, and both live entirely in this
adapter (the OpenEnv env server stays UNMODIFIED upstream):

  1. Working directory. TB2 tasks live at a fixed /app; SWE-Rebench-V2 donor
     images put the repo at varying paths (/app, /testbed, /<repo-name>, ...).
     We auto-detect the repo root once (the same probe tests/test.sh uses) and
     cd the agent into it, so the agent edits the real checkout. Override with
     OPENENV_TASK_WORKDIR to force a fixed dir.
  2. Scoring. The donor verifier does not write /logs/verifier/reward.txt; it
     prints ``RESULT: PASSED`` / ``RESULT: FAILED`` and exits 0/1. We run it via
     a plain ``exec`` step and derive the reward from that verdict line.

Env vars:
  OPENENV_ENV_URL    base_url of the env server (default: http://localhost:8003).
  OPENENV_MAX_TURNS  multi-turn cap (default: 30)
  OPENENV_MESSAGE_TIMEOUT_S  per-message WS recv timeout (default: 600; docker-mode
                     reset/exec/verify routinely exceed the client default of 60)
  OPENENV_MAX_ROLLOUT_TIME_SECONDS  hard wall-clock cap for one episode (default:
                     3600). An episode that does not return within the limit is
                     terminated and scored reward 0 (bounds long-trajectory
                     stragglers that would otherwise stall the whole rollout batch).
  AGENT_MODEL_NAME   model name sent to the policy (default: "model")
  MILES_ROUTER_EXTERNAL_HOST  optional host rewrite for off-cluster agents
  OPENENV_TASK_WORKDIR  force a fixed container dir for every agent command
                     (default: empty -> auto-detect the repo root per task).
  OPENENV_CONDA_ENV  conda env to activate for every agent command so the agent's
                     dev/test loop matches the grader (default: "testbed", the env
                     SWE-bench images install the repo + pytest into). Set to ""
                     to disable; probe is a no-op on images without such an env.
  OPENENV_SWEBENCH_TESTS_SRC  where the upstream env stages the task's tests inside
                     the container (default: /task/tests); copied to /tests for test.sh.
  OPENENV_EVAL_CMD   override the whole grading command (must still print
                     "<_REWARD_MARKER><float>" as its last stdout line).
"""

import asyncio
import logging
import os
import random
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse, urlunparse

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Env slots are finite: hosted spaces admit one session at a time
# (CAPACITY_REACHED) and the docker-mode server caps concurrent envs
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

# --- Working directory: auto-detect the repo root ----------------------------
# The env server execs commands in /task (a copy of the task *source*), but the
# code the agent must edit lives in the git repo baked into the image, whose path
# varies per SWE-Rebench-V2 donor (/app, /testbed, /<repo-name>, ...). We probe
# for it once (mirroring tests/test.sh: try /app and /testbed, then a shallow
# .git scan) and cache the answer, so every subsequent turn cds there cheaply.
# Setting OPENENV_TASK_WORKDIR forces a fixed dir and skips detection.
_TASK_WORKDIR = os.getenv("OPENENV_TASK_WORKDIR", "")
_REPO_ROOT_CACHE = "/tmp/.openenv_swebench_repo_root"
_DETECT_REPO_ROOT = (
    f'R=$(cat {_REPO_ROOT_CACHE} 2>/dev/null); '
    'if [ -z "$R" ]; then '
    'for c in /app /testbed; do [ -d "$c/.git" ] && R="$c" && break; done; '
    '[ -z "$R" ] && R=$(find / -maxdepth 3 -type d -name .git 2>/dev/null '
    '| grep -v node_modules | head -1 | xargs -r dirname); '
    'R=${R:-/}; '
    f'echo "$R" > {_REPO_ROOT_CACHE}; fi; '
)

# --- Conda env: run agent commands in the graded environment ------------------
# SWE-bench images install the repo (editable) plus pytest and every runtime dep
# into a conda env (default name "testbed"), NOT the base env. tests/test.sh
# activates it before grading, but the base env the container boots into lacks
# pytest and cannot even import the target package. Without activating it here,
# the agent runs blind -- `pytest` is not found and `import <pkg>` raises
# ModuleNotFoundError -- so it gets no test feedback and its edits are never the
# thing that gets graded. We activate the env (mirroring test.sh) for every agent
# command so the agent's dev/test loop matches the grader. The probe is a no-op
# on donor images that have no such env, so it stays safe for non-SWE-bench pools.
# Set OPENENV_CONDA_ENV="" to disable, or to another name to override "testbed".
_CONDA_ENV = os.getenv("OPENENV_CONDA_ENV", "testbed")
_ACTIVATE_ENV = (
    (
        'for _p in /opt/miniconda3 /opt/conda "$HOME/miniconda3" "$HOME/anaconda3"; do '
        f'if [ -f "$_p/bin/activate" ] && [ -d "$_p/envs/{_CONDA_ENV}" ]; then '
        f'. "$_p/bin/activate" {_CONDA_ENV} 2>/dev/null || true; break; fi; done; '
    )
    if _CONDA_ENV
    else ""
)

# --- Scoring: SWE-Rebench-V2 donor verifier ----------------------------------
# The donor verifier (tests/test.sh) self-locates the repo, applies the variant's
# test_patch, runs test_command from tests/config.json, writes output.json, and
# prints "RESULT: PASSED" (exit 0) or "RESULT: FAILED" (exit 1). It never writes
# /logs/verifier/reward.txt, so -- unlike the TB2 adapter -- we derive the reward
# from the verdict line: PASSED -> 1.0, FAILED -> 0.0, and neither present -> no
# recoverable reward (the verifier crashed before scoring, e.g. it could not find
# the repo), which the caller drops rather than scoring a false 0.0.
_TESTS_SRC = os.getenv("OPENENV_SWEBENCH_TESTS_SRC", "/task/tests")
_REWARD_MARKER = "__SB_REWARD__:"
# test.sh's exit code, echoed on its own marker purely for diagnostics.
_TESTSH_RC_MARKER = "__SB_TESTSH_RC__:"
_CANONICAL_EVAL_CMD = os.getenv("OPENENV_EVAL_CMD") or (
    "mkdir -p /tests /logs/verifier && "
    # test.sh reads /tests/config.json by absolute path, so stage the tests there.
    f"cp -a {_TESTS_SRC}/. /tests/ 2>/dev/null || true; "
    "bash /tests/test.sh > /tmp/sb_testsh.log 2>&1; rc=$?; "
    f"echo {_TESTSH_RC_MARKER}$rc; "
    "V=$(grep -aoE 'RESULT: (PASSED|FAILED)' /tmp/sb_testsh.log | tail -1); "
    f'if echo "$V" | grep -q PASSED; then echo {_REWARD_MARKER}1.0; '
    f'elif echo "$V" | grep -q FAILED; then echo {_REWARD_MARKER}0.0; '
    # No verdict line: verifier crashed before scoring -> emit empty value so the
    # caller recognizes an infra/harness failure and drops the sample.
    f'else echo {_REWARD_MARKER}; fi'
)


def _apply_workdir(command: str) -> str:
    """Prefix an agent command so it runs in the task's repo root."""
    if _TASK_WORKDIR:
        return f"{_ACTIVATE_ENV}cd {_TASK_WORKDIR} && {command}"
    return f'{_DETECT_REPO_ROOT}{_ACTIVATE_ENV}cd "$R" && {command}'


def _parse_reward_marker(output: str) -> float | None:
    """Parse the reward value the eval exec echoed on its marker line.

    Returns None when no reward can be recovered -- no marker line, an empty
    value (the verifier printed no RESULT verdict), or a non-numeric value.
    These are infra/harness failures, not a task the agent legitimately failed,
    so the caller drops the sample rather than scoring a false 0.0 that would
    pollute the training signal. A genuine failure emits 0.0 and is returned so.
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


# Per-message WS recv timeout. Docker-mode reset (container create), exec, and
# verify each routinely exceed the EnvClient default of 60s.
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

    The docker-mode server caps concurrent envs; over that cap it either returns
    CAPACITY_REACHED or closes the WebSocket cleanly (ConnectionClosedOK). Both
    mean "retry once a slot frees up", not a genuine episode failure. Match the
    close exceptions by class name so the adapter need not import websockets.
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
# The OpenEnv env server is the generic tbench2 server (it pulls the task's
# docker_image, copies the task dir into the container, and exposes exec); the
# SWE-bench task family reuses it wholesale and differs only in the task dirs.
def _load_env() -> dict[str, Any]:
    from tbench2_env import Tbench2Action, Tbench2Env

    return {"env": Tbench2Env, "action": Tbench2Action}


_DEFAULT_ENV_URL = "http://localhost:8003"


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
    env_url: str,
    policy: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Agentic loop: reset(task) -> {policy -> exec -> feed output back} -> verify.

    The policy emits one shell command per turn (a ```bash block or the bare
    reply), executed in the task's repo root; the loop ends when the policy stops
    emitting a command, says TASK_COMPLETE, or hits OPENENV_MAX_TURNS. Scoring
    runs the donor verifier (tests/test.sh) via an ``exec`` step and derives the
    binary reward from its RESULT verdict (needs no OpenEnv-side changes).
    """
    action_cls = classes["action"]
    task_id = metadata.get("task_id") or metadata.get("task_name")
    max_turns = int(os.getenv("OPENENV_MAX_TURNS", "30"))

    async def body(env: Any) -> tuple[float | None, int, list[float], list[float], float, float, int | None]:
        # Per-turn wall-clock timings. gen_times[i] is turn i's policy generation
        # latency; tool_times[i] is turn i's env.step(exec) latency. reset_time and
        # eval_time bracket the one-off reset() and the final verify() env steps.
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
            step_result = await env.step(action_cls(action_type="exec", command=_apply_workdir(command)))
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
        eval_result = await env.step(action_cls(action_type="exec", command=_CANONICAL_EVAL_CMD))
        eval_time = time.monotonic() - t0
        eval_output = _obs_field(eval_result, "output")
        reward = _parse_reward_marker(eval_output)
        testsh_rc = _parse_testsh_rc(eval_output)

        # rm-hack: the env server (TB2_OUTPUT_DIR=/tmp/tbench2_env_runs) leaves a
        # per-episode trial dir under that path after every episode, which fills
        # the sandbox overlay disk and trips ENOSPC. One episode holds the sandbox
        # at a time, so it is safe to purge them here.
        #
        # BUT the same dir also holds repo_cache/ (TB2_CACHE_DIR defaults to
        # output_dir/repo_cache) -- the shared checkout that reset() clones once
        # and every later episode reads its task from. A blanket `rm -rf .../*`
        # wiped repo_cache too, collapsing effective concurrency and exploding step
        # time. Preserve repo_cache; delete only the ephemeral per-trial dirs.
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

    reward, turns, gen_times, tool_times, reset_time, eval_time, testsh_rc = await _with_env(
        classes["env"], env_url, body
    )
    total_gen_time = sum(gen_times)
    # non_generation_time = everything the rollout spent outside policy generation:
    # per-turn exec latency plus the one-off reset() and verify() env steps. Feeds
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
    """Run one OpenEnv SWE-bench-style episode via the trained policy."""
    request_kwargs = request_kwargs or {}
    metadata = metadata or {}

    classes = _load_env()
    session_url = _resolve_session_url(base_url)
    model_name = os.getenv("AGENT_MODEL_NAME", os.getenv("SWE_AGENT_MODEL_NAME", "model"))
    env_url = os.getenv("OPENENV_ENV_URL", _DEFAULT_ENV_URL)

    policy = AsyncOpenAI(base_url=session_url, api_key="EMPTY")
    messages = _extract_messages(prompt)

    try:
        # Hard wall-clock cap: cancel the episode if it overruns and score it 0.
        # wait_for cancels the coroutine, so any in-flight policy call / env.step
        # is interrupted and the env session is closed by _with_env's async-with
        # during cancellation cleanup.
        reward, agent_metrics = await asyncio.wait_for(
            _multi_turn(classes, env_url, policy, model_name, messages, request_kwargs, metadata),
            timeout=_MAX_ROLLOUT_TIME_S,
        )
    except asyncio.TimeoutError:
        logger.warning(f"OpenEnv swebench episode exceeded {_MAX_ROLLOUT_TIME_S:.0f}s; terminating with reward 0")
        # eval_report empty: the episode was cancelled before the verifier ever
        # ran, so there is no report to surface.
        return {
            "reward": 0.0,
            "exit_status": "timeout",
            "eval_report": {},
            "agent_metrics": {"timed_out": 1},
        }
    except Exception as e:
        logger.error(f"OpenEnv swebench episode failed: {e}", exc_info=True)
        return None
    finally:
        await policy.close()

    # No recoverable reward means the verifier never produced a verdict
    # (infra/harness failure, not a legitimate task failure). Drop the sample --
    # returning it as reward 0.0 would inject a false negative into training.
    if reward is None:
        logger.warning(
            "OpenEnv swebench episode produced no verdict "
            f"(test.sh exit code={agent_metrics.get('testsh_rc')}); "
            "infra/harness failure, dropping sample"
        )
        return None

    # eval_report is intentionally empty: the reward-marker protocol echoes back
    # only the scalar reward. The verifier's detailed output.json is written
    # inside the sandbox at /logs/verifier/output.json and is deliberately not
    # captured back to the trainer, which consumes only `reward`.
    return {
        "reward": reward,
        "exit_status": "completed",
        "eval_report": {},
        "agent_metrics": agent_metrics,
    }
