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
``run_episode`` — how an env comes into being (see the episode-wiring note
there). The image recipe lives in ``tb2_sandbox_recipe`` and its Daytona
materialization in ``tb2_sandbox_daytona``; the recipe bakes the installed
``tbench2_env`` package -- OpenEnv's Terminal-Bench-2 environment package --
into the image, so this variant needs the pinned tbench2_env install from
the README (canonical test.sh scoring and verifier-asset withholding built
into the server).

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

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import openenv_agent_function as oaf
import openenv_sandbox_common as common
import tb2_sandbox_daytona

logger = logging.getLogger(__name__)

# Each episode materializes the per-task image declaratively from the Image
# definition, read off the local TB2 checkout (OPENENV_TB2_TASKS_DIR); repeat
# creates hit Daytona's build cache, and no named snapshot is involved.
#
# The sandbox's env server is the tbench2_env baked by the recipe, installed
# per the README (at or after the huggingface/OpenEnv#1012 merge): canonical
# tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. The launcher preflight rejects an
# older install outright, and the shared agent loop's harness-marker guard
# backstops it per episode.
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

_get_create_sem = common.lazy_semaphore(_CREATE_CONCURRENCY)


def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because Daytona rate-limited it.

    The SDK normalizes HTTP 429 to DaytonaRateLimitError; the text match is a
    fallback for older SDKs and server messages that only surface as text
    (e.g. "ThrottlerException: Too Many Requests").
    """
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where daytona is already in sys.modules,
    # having just raised `exc`), and older SDKs lack the class entirely.
    try:
        from daytona.common.errors import DaytonaRateLimitError

        if isinstance(exc, DaytonaRateLimitError):
            return True
    except ImportError:  # pragma: no cover - only without the daytona SDK
        pass
    s = str(exc).lower()
    return "throttler" in s or "too many requests" in s or "429" in s


def _start_declarative(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    daytona = tb2_sandbox_daytona.make_daytona()
    sandbox, url = tb2_sandbox_daytona.create_task_sandbox(
        daytona,
        Path(tasks_dir) / task_id,
        command_timeout_s=_COMMAND_TIMEOUT_S,
        ready_timeout_s=_READY_TIMEOUT_S,
    )
    return (lambda: daytona.delete(sandbox)), url


async def _start_task_sandbox(task_id: str) -> tuple[Any, str]:
    """Create one sandbox for *task_id* with the env server running.

    Returns (close_fn, base_url); close_fn deletes the sandbox. The
    orchestration — cancel-safe create, process-wide throttling, backoff on
    Daytona rate limits — is the provider-blind skeleton in
    ``openenv_sandbox_common``; this leg contributes only its start hook and
    throttle classifier.
    """
    return await common.start_task_sandbox(
        task_id,
        os.getenv("OPENENV_TB2_TASKS_DIR", "").strip(),
        start_fn=lambda tid, tdir: _start_declarative(tid, tdir),
        is_throttle=_is_throttle_error,
        sem=_get_create_sem(),
        max_retries=_CREATE_MAX_RETRIES,
        backoff_base_s=_CREATE_BACKOFF_BASE_S,
        backoff_cap_s=_CREATE_BACKOFF_CAP_S,
        logger=logger,
        provider="Daytona",
    )


@asynccontextmanager
async def _episode_env(env_cls: Any, metadata: dict[str, Any]):
    """Yield a connected env client on a fresh sandbox; delete it after."""
    async with common.episode_env(env_cls, metadata, start=_start_task_sandbox, logger=logger) as env:
        yield env


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

    No post-episode hygiene: the sandbox is deleted when the episode ends.
    """
    return await oaf._multi_turn(
        oaf._load_tbench2(),
        policy,
        model_name,
        messages,
        request_kwargs,
        metadata,
        run_body=_sandbox_run_body,
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
