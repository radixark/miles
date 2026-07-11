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
                     Ignored when OPENENV_DAYTONA_SNAPSHOT is set.
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

Daytona pool (optional; takes precedence over OPENENV_ENV_URL when set):
  OPENENV_DAYTONA_SNAPSHOT      Daytona snapshot name to provision sandboxes from
  OPENENV_DAYTONA_POOL_SIZE     # of long-lived sandboxes to spawn (default: 8)
  OPENENV_DAYTONA_PORT          server port inside the sandbox (default: 8000)
  DAYTONA_API_KEY               Daytona API key (required when SNAPSHOT is set)
"""

import asyncio
import logging
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import websockets.exceptions
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
_CANONICAL_EVAL_CMD = (
    "mkdir -p /tests /logs/verifier && "
    f"cp -a {_TB2_TESTS_SRC}/. /tests/ 2>/dev/null || true; "
    f"cd {_TASK_WORKDIR or '/app'} && bash /tests/test.sh > /tmp/tb2_testsh.log 2>&1; "
    f"echo {_REWARD_MARKER}$(cat /logs/verifier/reward.txt 2>/dev/null)"
)


def _apply_workdir(command: str) -> str:
    """Prefix an agent command so it runs in the real task workdir (/app)."""
    if not _TASK_WORKDIR:
        return command
    return f"cd {_TASK_WORKDIR} && {command}"


def _parse_reward_marker(output: str) -> float:
    """Parse the reward.txt value the canonical-eval exec echoed on its marker line."""
    for line in output.splitlines()[::-1]:
        if _REWARD_MARKER in line:
            raw = line.split(_REWARD_MARKER, 1)[1].strip()
            try:
                return float(raw) if raw else 0.0
            except ValueError:
                return 0.0
    return 0.0

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


# ---- Daytona sandbox pool (per-process singleton) ---------------------------
#
# When OPENENV_DAYTONA_SNAPSHOT is set, the adapter provisions a pool of N
# long-lived Daytona sandboxes (one DaytonaProvider per sandbox) on first use
# and rotates episodes across them through an asyncio.Queue. Episodes acquire a
# slot, run reset() -> step() -> evaluate(), then release. Concurrency is capped
# by the queue depth, so a rollout fan-out of 64 episodes onto a 64-slot pool
# runs without queueing.
#
# When the env var is unset the legacy code path runs (single OPENENV_ENV_URL
# host shared by all episodes), preserving the off-cluster manual-URL flow.
_POOL_PROVISION_TIMEOUT_S = float(os.getenv("OPENENV_DAYTONA_PROVISION_TIMEOUT_S", "600"))
_POOL_READY_TIMEOUT_S = float(os.getenv("OPENENV_DAYTONA_READY_TIMEOUT_S", "300"))
# Daytona rate-limits sandbox creation (ThrottlerException: Too Many Requests).
# Firing every create in a large pool at once trips it and, with no backoff,
# fails the whole pool. Cap in-flight creates and retry throttled ones with
# jittered exponential backoff so a wide (e.g. 64-slot) pool still comes up.
_POOL_PROVISION_CONCURRENCY = int(os.getenv("OPENENV_DAYTONA_PROVISION_CONCURRENCY", "8"))
_POOL_PROVISION_MAX_RETRIES = int(os.getenv("OPENENV_DAYTONA_PROVISION_MAX_RETRIES", "8"))
_POOL_PROVISION_BACKOFF_BASE_S = float(os.getenv("OPENENV_DAYTONA_PROVISION_BACKOFF_BASE_S", "2.0"))
_POOL_PROVISION_BACKOFF_CAP_S = float(os.getenv("OPENENV_DAYTONA_PROVISION_BACKOFF_CAP_S", "30.0"))


def _is_connection_error(exc: BaseException) -> bool:
    # OpenEnv's EnvClient.connect normalizes every handshake failure (refused,
    # timeout, HTTP 502 / rejected websocket) to a builtin ConnectionError;
    # a mid-stream drop surfaces as websockets' ConnectionClosedError from
    # recv/send. ConnectionClosedOK (clean 1000 close) is deliberately excluded.
    return isinstance(exc, (ConnectionError, websockets.exceptions.ConnectionClosedError))


def _is_throttle_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "throttler" in s or "too many requests" in s or "429" in s


@dataclass
class _DaytonaSlot:
    provider: Any
    url: str


class _DaytonaPool:
    """Process-wide pool of provisioned Daytona sandboxes."""

    _instance: "_DaytonaPool | None" = None

    def __init__(self, snapshot: str, api_key: str, size: int, port: int = 8000):
        self._snapshot = snapshot
        self._api_key = api_key
        self._size = size
        self._port = port
        self._queue: asyncio.Queue[_DaytonaSlot] | None = None
        self._slots: list[_DaytonaSlot] = []
        self._init_lock = asyncio.Lock()
        self._replace_tasks: set[asyncio.Task] = set()

    @classmethod
    def maybe(cls) -> "_DaytonaPool | None":
        """Return the singleton pool if env vars say to use one, else None."""
        snapshot = os.getenv("OPENENV_DAYTONA_SNAPSHOT", "").strip()
        if not snapshot:
            return None
        if cls._instance is None:
            api_key = os.environ["DAYTONA_API_KEY"]
            size = int(os.getenv("OPENENV_DAYTONA_POOL_SIZE", "8"))
            port = int(os.getenv("OPENENV_DAYTONA_PORT", "8000"))
            cls._instance = cls(snapshot=snapshot, api_key=api_key, size=size, port=port)
        return cls._instance

    async def _ensure_provisioned(self) -> None:
        if self._queue is not None:
            return
        async with self._init_lock:
            if self._queue is not None:
                return
            logger.info(
                f"Provisioning {self._size} Daytona sandboxes from snapshot:{self._snapshot} "
                f"(concurrency={_POOL_PROVISION_CONCURRENCY})"
            )
            sem = asyncio.Semaphore(_POOL_PROVISION_CONCURRENCY)
            results = await asyncio.gather(
                *(self._spawn_one(i, sem) for i in range(self._size)),
                return_exceptions=True,
            )
            slots: list[_DaytonaSlot] = []
            for i, r in enumerate(results):
                if isinstance(r, BaseException):
                    logger.error(f"Daytona slot {i} failed to provision: {r}")
                    continue
                slots.append(r)
            if not slots:
                raise RuntimeError("Failed to provision any Daytona sandboxes")
            queue: asyncio.Queue[_DaytonaSlot] = asyncio.Queue(maxsize=len(slots))
            for slot in slots:
                queue.put_nowait(slot)
            self._queue = queue
            self._slots = slots
            logger.info(f"Daytona pool ready: {len(slots)} / {self._size} slots online")

    async def _spawn_one(self, idx: int, sem: asyncio.Semaphore) -> _DaytonaSlot:
        from openenv.core.containers.runtime.daytona_provider import DaytonaProvider

        def _start() -> tuple[Any, str]:
            provider = DaytonaProvider(api_key=self._api_key, auto_stop_interval=0)
            url = provider.start_container(image=f"snapshot:{self._snapshot}", port=self._port)
            provider.wait_for_ready(url, timeout_s=_POOL_READY_TIMEOUT_S)
            return provider, url

        attempt = 0
        while True:
            try:
                # Hold the semaphore only for the create attempt; release it
                # during backoff so other slots keep the pipeline full.
                async with sem:
                    provider, url = await asyncio.wait_for(
                        asyncio.to_thread(_start), timeout=_POOL_PROVISION_TIMEOUT_S
                    )
                logger.info(f"Daytona slot {idx}: {url}")
                return _DaytonaSlot(provider=provider, url=url)
            except Exception as e:
                if not _is_throttle_error(e) or attempt >= _POOL_PROVISION_MAX_RETRIES:
                    raise
                attempt += 1
                delay = min(
                    _POOL_PROVISION_BACKOFF_CAP_S,
                    _POOL_PROVISION_BACKOFF_BASE_S * (2 ** (attempt - 1)),
                ) * (0.5 + random.random())
                logger.warning(
                    f"Daytona slot {idx} throttled (attempt {attempt}/"
                    f"{_POOL_PROVISION_MAX_RETRIES}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    async def acquire(self) -> _DaytonaSlot:
        await self._ensure_provisioned()
        assert self._queue is not None
        return await self._queue.get()

    async def release(self, slot: _DaytonaSlot) -> None:
        assert self._queue is not None
        self._queue.put_nowait(slot)

    def replace_broken(self, slot: _DaytonaSlot) -> None:
        logger.warning(f"Daytona slot {slot.url} marked broken; replacing in background")
        task = asyncio.create_task(self._replace(slot))
        self._replace_tasks.add(task)
        task.add_done_callback(self._replace_tasks.discard)

    async def _replace(self, slot: _DaytonaSlot) -> None:
        try:
            await asyncio.to_thread(slot.provider.stop_container)
        except Exception as e:
            logger.warning(f"Failed to stop broken sandbox {slot.url}: {e}")
        if slot in self._slots:
            self._slots.remove(slot)
        try:
            new_slot = await self._spawn_one(-1, asyncio.Semaphore(1))
        except Exception as e:
            logger.error(f"Failed to replace broken Daytona slot ({slot.url}): {e}")
            return
        self._slots.append(new_slot)
        assert self._queue is not None
        self._queue.put_nowait(new_slot)
        logger.info(f"Daytona slot replaced: {slot.url} -> {new_slot.url}")

    async def teardown(self) -> None:
        # Cancel in-flight replacements first: a task still awaiting _spawn_one
        # would otherwise enqueue a sandbox after teardown and leak it.
        try:
            for task in list(self._replace_tasks):
                task.cancel()
            if self._replace_tasks:
                await asyncio.gather(*self._replace_tasks, return_exceptions=True)
        finally:
            for slot in self._slots:
                try:
                    await asyncio.to_thread(slot.provider.stop_container)
                except Exception as e:
                    logger.warning(f"Failed to stop sandbox {slot.url}: {e}")


async def _with_env(env_cls: Any, env_url: str, body: Callable[[Any], Any]) -> Any:
    """Open an env session and run ``body(env)``, retrying while a slot is busy.

    If OPENENV_DAYTONA_SNAPSHOT is set, the sandbox URL is checked out from a
    process-wide pool; otherwise the shared ``env_url`` is used directly.
    """
    pool = _DaytonaPool.maybe()
    if pool is not None:
        slot = await pool.acquire()
        broken = False
        try:
            async with env_cls(base_url=slot.url, message_timeout_s=_MESSAGE_TIMEOUT_S) as env:
                return await body(env)
        except BaseException as e:
            broken = _is_connection_error(e)
            raise
        finally:
            if broken:
                pool.replace_broken(slot)
            else:
                await pool.release(slot)

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
) -> tuple[float, dict[str, Any]]:
    """Agentic loop: reset(task) -> {policy -> exec -> feed output back} -> evaluate (tbench2).

    The policy emits one shell command per turn (a ```bash block or the bare
    reply), executed in the real task workdir (_TASK_WORKDIR, /app); the loop ends
    when the policy stops emitting a command, says TASK_COMPLETE, or hits
    OPENENV_MAX_TURNS. Scoring runs the task's canonical tests/test.sh via an
    ``exec`` step and parses /logs/verifier/reward.txt for the binary reward
    (faithful to Terminal-Bench-2, and needs no OpenEnv-side changes).
    """
    action_cls = classes["action"]
    task_id = metadata.get("task_id") or metadata.get("task_name")
    max_turns = int(os.getenv("OPENENV_MAX_TURNS", "30"))

    async def body(env: Any) -> tuple[float, int, list[float], list[float], float, float]:
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
            step_result = await env.step(
                action_cls(action_type="exec", command=_apply_workdir(command))
            )
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
        eval_result = await env.step(
            action_cls(action_type="exec", command=_CANONICAL_EVAL_CMD)
        )
        eval_time = time.monotonic() - t0
        reward = _parse_reward_marker(_obs_field(eval_result, "output"))

        # rm-hack: the tbench2 env server (TB2_OUTPUT_DIR=/tmp/tbench2_env_runs)
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

        return reward, turns, gen_times, tool_times, reset_time, eval_time

    reward, turns, gen_times, tool_times, reset_time, eval_time = await _with_env(
        classes["env"], env_url, body
    )
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
    env_url = os.getenv("OPENENV_ENV_URL", _DEFAULT_ENV_URL)

    policy = AsyncOpenAI(base_url=session_url, api_key="EMPTY")
    messages = _extract_messages(prompt)

    try:
        # Hard wall-clock cap: cancel the episode if it overruns and score it 0.
        # wait_for cancels the coroutine, so any in-flight policy call / env.step
        # is interrupted and the pooled sandbox slot is released by _with_env's
        # finally block during cancellation cleanup.
        reward, agent_metrics = await asyncio.wait_for(
            _multi_turn(
                classes, env_url, policy, model_name, messages, request_kwargs, metadata
            ),
            timeout=_MAX_ROLLOUT_TIME_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"OpenEnv tbench2 episode exceeded {_MAX_ROLLOUT_TIME_S:.0f}s; "
            "terminating with reward 0"
        )
        return {
            "reward": 0.0,
            "exit_status": "timeout",
            "eval_report": {},
            "agent_metrics": {"timed_out": 1},
        }
    except Exception as e:
        logger.error(f"OpenEnv tbench2 episode failed: {e}", exc_info=True)
        return None

    return {
        "reward": reward,
        "exit_status": "completed",
        "eval_report": {},
        "agent_metrics": agent_metrics,
    }
