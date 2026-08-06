"""E2B-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN sandbox on an E2B-compatible
provider — built from the task's OFFICIAL image plus an env server layer,
killed when the episode ends. The provider is whatever the E2B SDK is pointed
at: E2B Cloud, or a self-hosted AgentENV deployment
(https://github.com/kvcache-ai/AgentENV) via E2B_API_URL / E2B_SANDBOX_URL.

The agent loop and training wrapper live in ``openenv_agent_function``
(sibling module) and are reused unchanged; this module only supplies its own
``run_episode`` — how an env comes into being (see the episode-wiring note
there). The image recipe lives in ``tb2_sandbox_recipe`` and its E2B
materialization in ``tb2_sandbox_e2b``; like the Daytona leg, this variant
needs the pinned tbench2_env install from the README (canonical test.sh
scoring and verifier-asset withholding built into the server).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR       path to a terminal-bench-2 checkout. Templates
                    are built once per task under a recipe-digest alias
                    (first episode of a task pays the build; repeats
                    warm-start), or pre-baked via the tb2_sandbox_e2b CLI.
  E2B_API_KEY / E2B_API_KEY_FILE   key supply, mirroring the Daytona leg's
                    DAYTONA_API_KEY / _FILE contract (file default
                    ~/.config/e2b/api_key; launchers forward the PATH, never
                    the value). AgentENV accepts any non-empty key today.
  E2B_API_URL / E2B_SANDBOX_URL    endpoint overrides read by the SDK itself;
                    set both to target a self-hosted AgentENV.
  OPENENV_E2B_CREATE_CONCURRENCY   max in-flight sandbox creates (default 4).
  OPENENV_E2B_READY_TIMEOUT_S      server-ready wait per sandbox (default 300).
  OPENENV_E2B_THROTTLE_PATTERNS    extra comma-separated lowercase substrings
                    classified as retryable throttling/capacity errors —
                    self-hosted providers surface "at capacity" differently
                    than E2B Cloud's 429s; extend without a code change.
  TB2_COMMAND_TIMEOUT_S            per-exec timeout inside the sandbox (default 900).
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import openenv_agent_function as oaf
import openenv_sandbox_common as common
import tb2_sandbox_e2b

logger = logging.getLogger(__name__)

# The sandbox's env server is the tbench2_env baked by the recipe, installed
# per the README (at or after the huggingface/OpenEnv#1012 merge): canonical
# tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. The launcher preflight rejects an
# older install outright, and the shared agent loop's harness-marker guard
# backstops it per episode.
#
# Providers rate-limit or run out of capacity under a fanned-out rollout; cap
# in-flight creates process-wide and retry throttled ones with jittered
# exponential backoff.
_CREATE_CONCURRENCY = int(os.getenv("OPENENV_E2B_CREATE_CONCURRENCY", "4"))
_CREATE_MAX_RETRIES = int(os.getenv("OPENENV_E2B_CREATE_MAX_RETRIES", "8"))
_CREATE_BACKOFF_BASE_S = float(os.getenv("OPENENV_E2B_CREATE_BACKOFF_BASE_S", "2.0"))
_CREATE_BACKOFF_CAP_S = float(os.getenv("OPENENV_E2B_CREATE_BACKOFF_CAP_S", "30.0"))
_READY_TIMEOUT_S = float(os.getenv("OPENENV_E2B_READY_TIMEOUT_S", "300"))
_COMMAND_TIMEOUT_S = int(os.getenv("TB2_COMMAND_TIMEOUT_S", "900"))

_get_create_sem = common.lazy_semaphore(_CREATE_CONCURRENCY)


def _extra_throttle_patterns() -> tuple[str, ...]:
    raw = os.getenv("OPENENV_E2B_THROTTLE_PATTERNS", "")
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because the provider throttled it.

    The SDK types HTTP 429 as RateLimitException; the text match covers
    proxies and self-hosted servers whose limits only surface as text, and
    OPENENV_E2B_THROTTLE_PATTERNS extends the set for provider-specific
    capacity errors (e.g. a full AgentENV node pool) without a code change.
    """
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where e2b is already in sys.modules,
    # having just raised `exc`), and offline tests must not require the SDK.
    try:
        from e2b.exceptions import RateLimitException

        if isinstance(exc, RateLimitException):
            return True
    except ImportError:  # pragma: no cover - only without the e2b SDK
        pass
    s = str(exc).lower()
    if "too many requests" in s or "429" in s or "rate limit" in s:
        return True
    return any(p in s for p in _extra_throttle_patterns())


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    sandbox, url = tb2_sandbox_e2b.create_task_sandbox(
        Path(tasks_dir) / task_id,
        command_timeout_s=_COMMAND_TIMEOUT_S,
        ready_timeout_s=_READY_TIMEOUT_S,
    )
    return (lambda: tb2_sandbox_e2b.kill_sandbox(sandbox)), url


async def _start_task_sandbox(task_id: str) -> tuple[Any, str]:
    """Create one sandbox for *task_id* with the env server running.

    Returns (close_fn, base_url); close_fn kills the sandbox. The
    orchestration — cancel-safe create, process-wide throttling, backoff on
    provider rate limits — is the provider-blind skeleton in
    ``openenv_sandbox_common``; this leg contributes only its start hook and
    throttle classifier.
    """
    return await common.start_task_sandbox(
        task_id,
        os.getenv("OPENENV_TB2_TASKS_DIR", "").strip(),
        start_fn=lambda tid, tdir: _start_sandbox(tid, tdir),
        is_throttle=_is_throttle_error,
        sem=_get_create_sem(),
        max_retries=_CREATE_MAX_RETRIES,
        backoff_base_s=_CREATE_BACKOFF_BASE_S,
        backoff_cap_s=_CREATE_BACKOFF_CAP_S,
        logger=logger,
        provider="E2B",
    )


@asynccontextmanager
async def _episode_env(env_cls: Any, metadata: dict[str, Any]):
    """Yield a connected env client on a fresh sandbox; kill it after."""
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
    """One episode in its own E2B sandbox, with the caller's own policy.
    Direct-drive entry, same contract as openenv_agent_function's.

    No post-episode hygiene: the sandbox is killed when the episode ends.
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
    """Run one OpenEnv tbench2 episode in its own E2B sandbox."""
    return await oaf._run_for_training(base_url, prompt, request_kwargs, metadata, run_episode)
