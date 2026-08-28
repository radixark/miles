"""Shared per-episode sandbox orchestration for the backend agent functions.

A backend module (``openenv_daytona_agent_function``,
``openenv_e2b_agent_function``, ``openenv_modal_agent_function``)
owns only what is genuinely provider-specific: how ONE sandbox with the env
server comes into being (its ``tb2_sandbox_*`` materialization module), which
errors count as retryable throttling, and its env-var knobs. Everything that
makes per-episode sandboxes safe under a fanned-out rollout is provider-blind
and lives here once:

  ``create_once``          cancel-safe create. ``asyncio.to_thread`` is not
                cancellable: when an episode's wall-clock cap cancels the
                coroutine mid-create, the worker thread keeps running and its
                (close_fn, url) result would be discarded — leaking a sandbox
                until the provider-side TTL backstop reclaims it. The result
                is recorded thread-side and, on cancellation, handed to a
                reaper that closes the orphan promptly once the create
                finishes.
  ``lazy_semaphore``       the create-throttle semaphore each backend passes in,
                built on first use rather than at import.
  ``SandboxBackend``       the orchestration itself. ``start_task_sandbox``
                throttles creates process-wide (the semaphore) and retries the
                errors the backend classifies as throttling with jittered
                exponential backoff — anything else propagates immediately, and
                the semaphore is held only for the create attempt and released
                during backoff so other episodes keep the pipeline full.
                ``episode_env`` is the async context manager mirroring one
                episode's lifetime: fresh sandbox -> connected env client ->
                close.
"""

import asyncio
import importlib
import logging
import os
import random
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from miles.rollout.agent_function import InfraAbort

# A provider's "start one sandbox" hook: (task_id, tasks_dir) -> (close_fn, base_url).
StartFn = Callable[[str, str], tuple[Callable[[], None], str]]

# The canonical per-episode backend registry — the launcher and the sibling
# tools (scan_golden, eval_tbench2_via_api) all resolve through here, so the
# accepted names and the "agentenv is the e2b backend" aliasing cannot drift.
# Adding a provider is one entry plus its two modules (backend + materialization).
AGENT_MODULES = {
    "daytona": "openenv_daytona_agent_function",
    "e2b": "openenv_e2b_agent_function",
    "modal": "openenv_modal_agent_function",
}
AGENT_FUNCTIONS = {backend: f"{module}.run" for backend, module in AGENT_MODULES.items()}
_ALIASES = {"agentenv": "e2b"}


def backend_names() -> str:
    """The accepted names, for help text that must not drift from the registry."""
    return ", ".join(sorted([*AGENT_FUNCTIONS, *_ALIASES]))


def resolve_backend(name: str | None) -> str:
    """Normalize a backend name; reject unknown and missing ones alike.

    There is deliberately no default. Which provider runs decides whose quota
    an entire rollout spends and which credentials must be present, so a name
    left out is a question to answer rather than one to guess at.
    """
    backend = (name or "").strip().lower()
    allowed = backend_names()
    if not backend:
        raise ValueError(f"no sandbox backend named; choose one of: {allowed}")
    backend = _ALIASES.get(backend, backend)
    if backend not in AGENT_FUNCTIONS:
        raise ValueError(f"unknown sandbox backend {name!r}; choose one of: {allowed}")
    return backend


def load_backend(name: str | None) -> "SandboxBackend":
    """The SandboxBackend named by *name*, imported on demand.

    The sibling tools (scan_golden, eval_tbench2_via_api) drive one backend
    directly instead of through the training entry point. Resolving it here
    means they gain a provider by naming it, not by growing an import branch
    each — and a provider's SDK is imported only when it is the one selected.
    """
    return importlib.import_module(AGENT_MODULES[resolve_backend(name)]).BACKEND


def lazy_semaphore(limit: int) -> Callable[[], asyncio.Semaphore]:
    """Return a getter for a *limit*-slot semaphore created on first call.

    A backend reads its concurrency knob at import, but a semaphore constructed
    then would belong to whatever loop happens to be current — the rollout
    loop does not exist yet. Deferring construction to the first episode ties
    it to the loop that actually awaits on it.
    """
    holder: list[asyncio.Semaphore] = []

    def get() -> asyncio.Semaphore:
        if not holder:
            holder.append(asyncio.Semaphore(limit))
        return holder[0]

    return get


async def create_once(start_fn: StartFn, task_id: str, tasks_dir: str, *, logger: logging.Logger) -> tuple[Any, str]:
    """One sandbox-create attempt, safe against cancellation mid-create."""
    result: list[tuple[Any, str]] = []
    done = threading.Event()

    def _start() -> tuple[Any, str]:
        try:
            result.append(start_fn(task_id, tasks_dir))
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
                    logger.info(f"Closed sandbox orphaned by cancelled episode: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to close orphaned sandbox for {task_id}: {e}")

        threading.Thread(target=_reap, name=f"tb2-sandbox-reap-{task_id}", daemon=True).start()
        raise


def _agent_function():
    """``openenv_agent_function``, imported on use rather than at module scope.

    The one lazy import in this module, and the reason is the module's other
    callers: a provider CLI (``python tb2_sandbox_e2b.py ...``) and the
    launcher reach this module for the backend registry alone, and neither
    should have to have the agent loop's dependencies (openai) installed to
    resolve a backend name.
    """
    import openenv_agent_function

    return openenv_agent_function


# Throttle text every provider shares: an HTTP 429 surfaced as a message
# rather than a typed error. A backend adds its own vocabulary (Daytona's
# "throttler", Modal's "resource exhausted").
_THROTTLE_TEXT = ("too many requests", "429", "rate limit")


def throttle_text(exc: BaseException, *extra_patterns: str, patterns_env_var: str | None = None) -> bool:
    """True when *exc*'s message reads like throttling or exhausted capacity.

    The typed-exception half of the decision stays in the backend (only it knows
    its SDK's class); this is the half that is identical everywhere, plus the
    backend's *extra_patterns*.

    *patterns_env_var* extends the set at runtime, and exists for exactly one
    case: an endpoint whose capacity errors are worded by whoever deployed it,
    which is why only the e2b backend (self-hosted AgentENV) passes one.
    """
    s = str(exc).lower()
    if any(p in s for p in (*_THROTTLE_TEXT, *(p.lower() for p in extra_patterns))):
        return True
    raw = os.getenv(patterns_env_var, "") if patterns_env_var else ""
    return any(p in s for p in (chunk.strip().lower() for chunk in raw.split(",")) if p)


# Backoff shape for throttled creates: jittered exponential, shared by every
# backend. Constants rather than knobs — a rollout tunes how MANY creates are in
# flight (create_concurrency) and how long to keep trying (max_retries), not the
# curve between them.
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0

# Knob defaults shared by every backend, read from its OWN prefix so one
# provider can be re-tuned without touching another.
_KNOB_DEFAULTS = {
    "create_concurrency": ("CREATE_CONCURRENCY", 4, int),
    "max_retries": ("CREATE_MAX_RETRIES", 8, int),
}


def backend_knobs(env_prefix: str) -> dict[str, Any]:
    """The create-throttling knobs for a backend, read from OPENENV_<PREFIX>_*.

    Read once, at the importing module's import time: a rollout worker is a
    fresh process per run, so a knob changed mid-run would be a knob that
    silently did nothing.
    """
    return {
        field: cast(os.getenv(f"OPENENV_{env_prefix}_{suffix}", str(default)))
        for field, (suffix, default, cast) in _KNOB_DEFAULTS.items()
    }


@dataclass
class SandboxBackend:
    """The provider-blind half of a per-episode sandbox agent function.

    Every backend is the same rollout-facing object — throttled cancel-safe
    creates, an env client bound to one episode's sandbox, the training and
    direct-drive entry points — around two provider-specific holes:
    ``start_sandbox`` (how ONE sandbox with the env server comes into being)
    and ``is_throttle`` (which of its SDK's errors are worth retrying). A backend
    module supplies those two, its knobs, and its documentation; nothing else
    about it can drift from its siblings because nothing else lives there.

    The fields are the seams: tests substitute ``start_sandbox`` or
    ``episode_env`` and shrink the backoff on the instance, so the knobs stay
    patchable without a backend re-reading its environment.
    """

    provider: str
    start_sandbox: StartFn
    is_throttle: Callable[[BaseException], bool]
    logger: logging.Logger
    create_concurrency: int = _KNOB_DEFAULTS["create_concurrency"][1]
    max_retries: int = _KNOB_DEFAULTS["max_retries"][1]
    backoff_base_s: float = BACKOFF_BASE_S
    backoff_cap_s: float = BACKOFF_CAP_S

    def __post_init__(self) -> None:
        self._get_sem = lazy_semaphore(self.create_concurrency)

    async def start_task_sandbox(self, task_id: str) -> tuple[Any, str]:
        """Create one sandbox for *task_id* with the env server running.

        Returns (close_fn, base_url); close_fn releases the sandbox. Creation
        is throttled process-wide (the create semaphore) and retried with
        jittered exponential backoff on throttle errors.

        A create that still fails -- throttled past the retry budget, or any
        other error -- raises InfraAbort: the policy has not acted yet, so the
        sample is discarded rather than trained on as a 0.
        """
        tasks_dir = os.getenv("OPENENV_TB2_TASKS_DIR", "").strip()
        attempt = 0
        while True:
            try:
                async with self._get_sem():
                    return await create_once(self.start_sandbox, task_id, tasks_dir, logger=self.logger)
            except Exception as e:
                if not self.is_throttle(e) or attempt >= self.max_retries:
                    raise InfraAbort(
                        "SandboxUnavailable", f"{self.provider} sandbox create failed for {task_id}: {e}"
                    ) from e
                attempt += 1
                delay = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1))) * (0.5 + random.random())
                self.logger.warning(
                    f"{self.provider} create throttled for {task_id} "
                    f"(attempt {attempt}/{self.max_retries}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    @asynccontextmanager
    async def episode_env(self, env_cls: Any, metadata: dict[str, Any]):
        """Yield a connected env client on a fresh sandbox; release it after."""
        oaf = _agent_function()
        task_id = metadata.get("task_id") or metadata.get("task_name")
        if not task_id:
            raise ValueError("the sandbox is built for one task: metadata['task_id'] is required")
        close_fn, url = await self.start_task_sandbox(str(task_id))
        try:
            async with env_cls(base_url=url, message_timeout_s=oaf.MESSAGE_TIMEOUT_S) as env:
                yield env
        finally:
            try:
                await asyncio.to_thread(close_fn)
            except Exception as e:
                self.logger.warning(f"Failed to close sandbox for {task_id}: {e}")

    async def _run_body(self, env_cls: Any, metadata: dict[str, Any], body: Any) -> Any:
        # self.episode_env, not the module function: an instance attribute
        # substituted by a test must win here.
        async with self.episode_env(env_cls, metadata) as env:
            return await body(env)

    async def run_episode(
        self,
        policy: Any,
        model_name: str,
        messages: list[dict[str, str]],
        request_kwargs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[float | None, dict[str, Any]]:
        """One episode in its own sandbox, with the caller's own policy.
        Direct-drive entry, same contract as openenv_agent_function's.

        No post-episode hygiene: the sandbox is released when the episode ends.
        """
        oaf = _agent_function()
        return await oaf.multi_turn(
            oaf.load_tbench2(),
            policy,
            model_name,
            messages,
            request_kwargs,
            metadata,
            run_body=self._run_body,
        )

    async def run(
        self,
        base_url: str,
        prompt: Any,
        request_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Run one OpenEnv tbench2 episode in its own sandbox (training entry)."""
        oaf = _agent_function()
        return await oaf.run_for_training(base_url, prompt, request_kwargs, metadata, self.run_episode)
