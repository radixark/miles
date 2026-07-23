"""Daytona-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN Daytona cloud sandbox — built
from the task's OFFICIAL image plus an env server layer, deleted when the
episode ends. Full per-task image fidelity with zero shared infrastructure
(no Docker host, no long-lived env server) and zero cross-episode state
leakage.

The agent loop and training wrapper live in ``openenv_agent_function``
(sibling module) and are reused unchanged; this module only supplies its own
``run_episode`` — how an env comes into being and which server contract it
speaks (``native_evaluate``, see the episode-wiring note there). The image
recipe lives in ``tb2_sandbox_recipe`` and its Daytona materialization in
``tb2_sandbox_daytona``; the recipe bakes the installed ``tbench2_env``
package -- OpenEnv's Terminal-Bench-2 environment package -- into the image,
so this variant needs the pinned tbench2_env install from the README
(canonical test.sh scoring and verifier-asset withholding built into the
server).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR        path to a terminal-bench-2 checkout: build the
                     sandbox declaratively per episode. Daytona caches image
                     layers by definition hash, so only the first episode of a
                     task builds (~10 min); repeats start in ~1 min. No named
                     snapshots, so no org snapshot quota.
  DAYTONA_API_KEY              the Daytona API key, authenticating every
                     sandbox create/delete. Read from the worker's own
                     node-local environment; nothing forwards it. Supply it
                     via platform-injected pod env, or by exporting it in
                     the shell that starts ray on a single host.
  DAYTONA_API_KEY_FILE         fallback when DAYTONA_API_KEY is unset: path
                     of a file holding the key (default
                     ~/.config/daytona/api_key). Launchers forward this path
                     instead of the key itself, because ray runtime_env is
                     logged in plaintext. Point it at a file every node can
                     read: a dotfile, K8s Secret mount, or shared-FS path.
  OPENENV_DAYTONA_CREATE_CONCURRENCY  max in-flight sandbox creates (default 4).
  OPENENV_DAYTONA_READY_TIMEOUT_S     server-ready wait per sandbox (default 300).
  TB2_COMMAND_TIMEOUT_S        per-exec timeout inside the sandbox (default 900).
"""

import asyncio
import logging
import os
import random
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import openenv_agent_function as oaf

logger = logging.getLogger(__name__)

# Each episode materializes the per-task image declaratively from the Image
# definition, read off the local TB2 checkout (OPENENV_TB2_TASKS_DIR); repeat
# creates hit Daytona's build cache, and no named snapshot is involved.
#
# The sandbox's env server is the CURRENT upstream tbench2_env baked by the
# recipe — carrying the fixes upstreamed via huggingface/OpenEnv#965 + #972:
# canonical tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. So run_episode here sets
# native_evaluate=True and the adapter-side compensation machinery in
# openenv_agent_function (_apply_workdir / _CANONICAL_EVAL_CMD) is deliberately
# not applied — the launcher preflight rejects an older install outright.
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


def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because Daytona rate-limited it.

    The daytona SDK is a lazy dependency of this module only (shared-server
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
    import tb2_sandbox_daytona

    daytona = tb2_sandbox_daytona.make_daytona()
    sandbox, url = tb2_sandbox_daytona.create_task_sandbox(
        daytona,
        Path(tasks_dir) / task_id,
        command_timeout_s=_COMMAND_TIMEOUT_S,
        ready_timeout_s=_READY_TIMEOUT_S,
    )
    return (lambda: daytona.delete(sandbox)), url


async def _create_once(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    """One sandbox-create attempt, safe against cancellation mid-create.

    asyncio.to_thread is not cancellable: when the episode's wall-clock cap
    cancels this coroutine mid-create, the worker thread keeps running and its
    (close_fn, url) result would be discarded — leaking a sandbox that would
    otherwise run (and bill) until the recipe's TTL backstop reclaims it.
    Record the result thread-side and, on cancellation, hand it to a reaper
    that deletes the orphan promptly once the create finishes.
    """
    result: list[tuple[Any, str]] = []
    done = threading.Event()

    def _start() -> tuple[Any, str]:
        try:
            result.append(_start_declarative(task_id, tasks_dir))
        finally:
            done.set()
        return result[0]

    try:
        return await asyncio.to_thread(_start)
    except asyncio.CancelledError:

        def _reap() -> None:
            done.wait()
            for close_fn, _url in result:
                try:
                    close_fn()
                    logger.info(f"Deleted sandbox orphaned by cancelled episode: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete orphaned sandbox for {task_id}: {e}")

        threading.Thread(target=_reap, name=f"tb2-sandbox-reap-{task_id}", daemon=True).start()
        raise


async def _start_task_sandbox(task_id: str) -> tuple[Any, str]:
    """Create one sandbox for *task_id* with the env server running.

    Returns (close_fn, base_url); close_fn deletes the sandbox. Creation is
    throttled process-wide and retried on Daytona rate limits.
    """
    tasks_dir = os.getenv("OPENENV_TB2_TASKS_DIR", "").strip()

    attempt = 0
    while True:
        try:
            # Hold the semaphore only for the create attempt; release it during
            # backoff so other episodes keep the pipeline full.
            async with _get_create_sem():
                return await _create_once(task_id, tasks_dir)
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
    """Yield a connected env client on a fresh sandbox; delete it after."""
    task_id = metadata.get("task_id") or metadata.get("task_name")
    if not task_id:
        raise ValueError("the sandbox is built for one task: metadata['task_id'] is required")
    close_fn, url = await _start_task_sandbox(str(task_id))
    try:
        async with env_cls(base_url=url, message_timeout_s=oaf._MESSAGE_TIMEOUT_S) as env:
            yield env
    finally:
        try:
            await asyncio.to_thread(close_fn)
        except Exception as e:
            logger.warning(f"Failed to delete sandbox for {task_id}: {e}")


async def _sandbox_run_body(env_cls: Any, metadata: dict[str, Any], body: Any) -> Any:
    """Run *body* on a fresh sandbox for the episode's task."""
    async with _episode_env(env_cls, metadata) as env:
        return await body(env)


async def run_episode(
    policy: Any,
    model_name: str,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """One episode in its own Daytona sandbox, with the caller's own
    policy. Direct-drive entry, same contract as openenv_agent_function's.

    native_evaluate=True: the baked server carries the OpenEnv#965/#972 fixes —
    raw exec commands (WORKDIR resolved server-side), scoring via the native
    `evaluate` action. No post-episode hygiene: the sandbox is deleted when
    the episode ends.
    """
    return await oaf._multi_turn(
        oaf._load_tbench2(),
        policy,
        model_name,
        messages,
        request_kwargs,
        metadata,
        run_body=_sandbox_run_body,
        native_evaluate=True,
    )


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one OpenEnv tbench2 episode in its own Daytona sandbox."""
    return await oaf._run_for_training(base_url, prompt, request_kwargs, metadata, run_episode)
