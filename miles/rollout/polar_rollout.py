"""Miles-side async rollout bridge for Polar-managed agent sessions.

Faithful port of ProRL-Agent-Server's ``slime_bridge.rollout`` adapted for the
Miles rollout package. Single entrypoint ``generate_rollout_polar_async``
routes training to a persistent background worker and evaluation to a one-shot
submit+poll batch. Both paths speak Polar's async-only HTTP surface
(``/rollout/task/submit`` + ``/rollout/task/{task_id}``).

The Polar ``slime_bridge``-provided helpers are imported from the sibling
Miles modules (:mod:`.polar_config`, :mod:`.polar_adapter`) rather than
re-implemented here. Polar core types (``polar.rollout.models.TaskResult`` /
``TaskStatus``) are referenced only under ``TYPE_CHECKING`` and resolved lazily
at their point of use, so this module imports cleanly under a plain Miles
environment with no ``polar`` package installed. The optional runtime deps
(``httpx``, ``fastapi``, ``uvicorn``) are likewise only required on the code
paths that actually talk to a Polar rollout server, never at import time.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import gzip
import json
import logging
import math
import os
import queue
import statistics
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .polar_adapter import RolloutLogprobError, session_result_to_samples
from .polar_config import (
    PolarSlimeConfig,
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
)

if TYPE_CHECKING:
    from polar.rollout.models import TaskResult

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0  # seconds between task-status polls (eval / no-callback path)
_CALLBACK_FALLBACK_POLL_SECONDS = 60.0  # defensive backstop for dropped callbacks
_LONGEST_TRACE_ARTIFACT_INTERVAL = 5  # dump longest trace every N rollouts


class PolarRolloutSchedulerError(RuntimeError):
    """Raised when the async Polar scheduler cannot safely make progress."""


class PolarLowCompleteAcceptFractionError(PolarRolloutSchedulerError):
    """Raised when a completed task has too few trainable completed sessions."""


@dataclass(slots=True)
class _DeferredGroup:
    group: list[Any]


@dataclass(slots=True)
class _PendingGroup:
    group_id: int
    group: list[Any]
    submitted_rollout_id: int
    policy_version: int
    session_cost: int


@dataclass(slots=True)
class _CompletedGroup:
    group_id: int
    group: list[Any]
    samples: list[Any]
    task_id: str
    submitted_rollout_id: int
    policy_version: int
    session_count: int
    completed_at: float = field(default_factory=time.monotonic)

# ---------------------------------------------------------------------------
# Global worker singleton
# ---------------------------------------------------------------------------
_global_async_worker: AsyncPolarRolloutWorker | None = None
_worker_lock = threading.Lock()


def get_global_async_worker(args: Any, data_source: Any) -> AsyncPolarRolloutWorker:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            _global_async_worker = AsyncPolarRolloutWorker(args, data_source)
            _global_async_worker.start()
        return _global_async_worker


def stop_global_worker() -> None:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.stop()
            _global_async_worker = None


# ---------------------------------------------------------------------------
# Lazily-resolved optional dependencies
# ---------------------------------------------------------------------------
def _load_task_result_type() -> Any:
    """Return the Polar ``TaskResult`` model, importing it lazily.

    Polar core is an optional, runtime-only dependency for this module (only
    reachable once a Polar rollout server is actually being driven). It is
    loaded here instead of at module import so the module stays importable
    under a plain Miles environment without ``polar`` installed. Raises a
    clear error if Polar is unavailable when a caller actually needs it.
    """
    try:
        from polar.rollout.models import TaskResult
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise ImportError(
            "Building a Polar TaskResult requires the 'polar' package, which is "
            "not installed or not importable in this environment."
        ) from exc
    return TaskResult


def _load_task_status_type() -> Any:
    """Return the Polar ``TaskStatus`` model, importing it lazily.

    Lazy for the same reason as :func:`_load_task_result_type`.
    """
    try:
        from polar.rollout.models import TaskStatus
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise ImportError(
            "Parsing Polar task status requires the 'polar' package, which is "
            "not installed or not importable in this environment."
        ) from exc
    return TaskStatus


def _load_httpx() -> Any:
    """Return the ``httpx`` module, importing it lazily.

    Only needed on the code paths that actually talk to a Polar rollout
    server; kept lazy so the module imports even in a stripped Miles env.
    """
    import httpx

    return httpx


# FastAPI resolves the string annotation against this module's globals. The
# import cannot live only inside the listener function: under postponed
# annotations FastAPI would instead treat ``Request`` as a required query
# parameter and reject JSON callback bodies with HTTP 422.
try:
    from fastapi import Request as _FastAPIRequest
except ImportError:  # pragma: no cover - fastapi is optional at import time
    _FastAPIRequest = Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
) -> dict[str, Any]:
    first_sample = group[0]
    prompt_text = _prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
    instruction = render_instruction(
        args=args,
        config=config,
        sample=first_sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )
    return render_task_payload(
        args=args,
        config=config,
        sample=first_sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )


def _attach_scheduler_metadata(
    payload: dict[str, Any],
    *,
    group_id: int,
    policy_version: int,
    rollout_step: int,
) -> None:
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("polar task metadata must be a mapping when provided")
    payload["metadata"] = {
        **metadata,
        "group_id": group_id,
        "policy_version": policy_version,
        "rollout_step": rollout_step,
    }


async def _submit_and_wait_for_task(
    client: Any,
    base_url: str,
    payload: dict[str, Any],
    *,
    poll_interval: float = _POLL_INTERVAL,
    task_timeout: float | None = None,
) -> TaskResult:
    """Submit one task and poll until terminal or the task deadline expires."""
    resp = await client.post(
        f"{base_url}/rollout/task/submit",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    httpx = _load_httpx()
    deadline = None if task_timeout is None else time.monotonic() + task_timeout
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"Polar task {task_id} did not reach a terminal state within "
                f"{task_timeout:g}s"
            )
        await asyncio.sleep(poll_interval)
        try:
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            logger.warning("Polling Polar task %s failed; continuing: %s", task_id, exc)
            continue
        status = _load_task_status_type().model_validate(status_resp.json())
        if status.status in ("completed", "failed"):
            break

    return _load_task_result_type()(
        task_id=task_id,
        status=status.status,
        results=status.results,
        result_paths=status.result_paths,
    )


def _resolve_max_tokens(args: Any) -> int | None:
    """Per-sample token cap Miles' dynamic batcher can fit on one GPU.

    Megatron asserts every sample length <= max_tokens_per_gpu * cp_size.
    Deep agent trajectories can exceed this (24-turn sessions → 80k+ tokens)
    and must be dropped before they reach the batcher.
    """
    mtpg = getattr(args, "max_tokens_per_gpu", None)
    if not mtpg:
        return None
    cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
    return int(mtpg) * cp_size


def _convert_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one task's session results into flat Miles samples.

    Each session → one trajectory → N traces → N samples, all tagged
    with the same ``Sample.index`` so the reward post-processor groups
    them as one trajectory. The index is taken from the originating
    group sample at matching position, falling back to the position
    within the task result.
    """
    group_index = _group_index_for(group)
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results):
        source = group[pos] if pos < len(group) else None
        traj_idx = int(getattr(source, "index", pos) if source is not None else pos)
        group_samples.extend(
            session_result_to_samples(
                session_result,
                group_index,
                trajectory_index=traj_idx,
                reward_key=config.reward_key,
                max_tokens=max_tokens,
            )
        )
    return group_samples


def _trainable_token_count(sample: Any) -> int:
    if bool(getattr(sample, "remove_sample", False)):
        return 0
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0)
    return sum(1 for value in loss_mask if int(value) != 0)


def _has_trainable_tokens(samples: list[Any]) -> bool:
    return any(_trainable_token_count(sample) > 0 for sample in samples)


def _low_complete_accept_fraction_rejection_reason(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    samples: list[Any],
) -> str | None:
    threshold = config.min_complete_accept_fraction
    if threshold <= 0.0:
        return None

    total_sessions = len(task_result.results)
    if total_sessions <= 0:
        return "empty task results"

    completed_trainable = _completed_trainable_session_count(task_result, samples)
    required = math.ceil(total_sessions * threshold)
    if completed_trainable >= required:
        return None

    fraction = completed_trainable / total_sessions
    return (
        f"completed trainable sessions {completed_trainable}/{total_sessions} "
        f"({fraction:.3f}) below polar_min_complete_accept_fraction={threshold:g} "
        f"(requires >= {required})"
    )


def _completed_trainable_session_count(task_result: TaskResult, samples: list[Any]) -> int:
    """Count sessions with trainable content, regardless of task-level labels.

    Polar may mark a task failed when one member session fails while other
    sessions in the prompt group completed. Admission should be based on usable
    per-session training content, not the aggregate task label.
    """
    trainable_session_ids: set[str] = set()
    for sample in samples:
        if _trainable_token_count(sample) <= 0:
            continue
        session_id = _sample_session_id(sample)
        if session_id:
            trainable_session_ids.add(session_id)

    result_session_ids = {
        str(result.session_id) for result in task_result.results if result.session_id
    }
    return len(trainable_session_ids & result_session_ids)


def _sample_session_id(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    session_id = polar_meta.get("session_id") or getattr(sample, "session_id", None)
    return str(session_id) if session_id else None


def _is_zero_trainable_error(exc: BaseException) -> bool:
    return "zero trainable tokens" in str(exc)


def _is_retriable_polar_task_error(exc: BaseException) -> bool:
    """Identify Polar task-store misses that are safe to resubmit."""
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 404:
        return True
    message = str(exc).lower()
    return (
        "404" in message
        or "retriable after restart" in message
        or ("task" in message and "not found" in message)
    )


def _annotate_accepted_samples(
    samples: list[Any],
    *,
    accepted_rollout_id: int,
    staleness: int,
    policy_version: int,
    scheduler_group_id: int,
) -> None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        polar_meta = metadata.setdefault("polar", {})
        if not isinstance(polar_meta, dict):
            polar_meta = {}
            metadata["polar"] = polar_meta
        polar_meta.update(
            {
                "accepted_rollout_id": int(accepted_rollout_id),
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
                "scheduler_group_id": int(scheduler_group_id),
            }
        )
        train_metadata = getattr(sample, "train_metadata", None)
        if train_metadata is None:
            train_metadata = {}
            sample.train_metadata = train_metadata
        train_metadata.update(
            {
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
            }
        )


# ---------------------------------------------------------------------------
# Persistent training worker
# ---------------------------------------------------------------------------
class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop. Pulls
    sample groups from ``data_source``, submits them to the async
    ``/rollout/task/submit`` endpoint, polls until completion, converts
    results, and pushes them into ``output_queue``. Training loops call
    ``drain_completed()`` to collect finished groups.
    """

    def __init__(self, args: Any, data_source: Any) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
        # Output queue is a handoff channel; the durable overflow buffer is
        # `_completed_buffer`, which is drained in bounded chunks by training.
        queue_maxsize = max(32, batch_size * self.config.max_async_level * 2)
        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue(maxsize=queue_maxsize)
        self.deferred_queue: queue.Queue[_DeferredGroup] = queue.Queue()
        self._completed_buffer: deque[_CompletedGroup] = deque()
        self._running = True
        self._thread: threading.Thread | None = None
        self._group_counter = 0
        self._batch_size = batch_size
        self._current_rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        self._requested_groups = 0
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.RLock()
        self._metrics: dict[str, float] = {}
        self._active_groups = 0
        self._active_sessions = 0
        self._completed_buffer_size = 0
        # Per-task callback plumbing: event fires when the rollout server POSTs
        # the terminal TaskResult to our local listener.
        self._task_events: dict[str, asyncio.Event] = {}
        self._task_results: dict[str, Any] = {}
        self._callback_url: str | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="polar-async-rollout")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- results ---------------------------------------------------------------

    def set_rollout_context(self, rollout_id: int) -> None:
        with self._state_lock:
            self._current_rollout_id = int(rollout_id)

    def request_groups(self, count: int) -> None:
        if count <= 0:
            return
        with self._state_lock:
            self._requested_groups += int(count)

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise PolarRolloutSchedulerError(str(self._fatal_error)) from self._fatal_error

    def drain_completed(
        self,
        *,
        max_groups: int,
        rollout_id: int,
    ) -> list[_CompletedGroup]:
        self.raise_if_failed()

        while True:
            try:
                self._completed_buffer.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        with self._state_lock:
            self._completed_buffer_size = len(self._completed_buffer)

        accepted: list[_CompletedGroup] = []
        while self._completed_buffer and len(accepted) < max_groups:
            completed = self._completed_buffer.popleft()
            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness > self.config.max_off_policy_steps:
                self._inc_metric("polar/stale_groups")
                reason = (
                    f"staleness {staleness} exceeded max_off_policy_steps="
                    f"{self.config.max_off_policy_steps}"
                )
                self._inc_metric("polar/dropped_groups")
                self._inc_metric("polar/dropped_stale_groups")
                self._inc_metric("polar/dropped_sessions", completed.session_count)
                logger.warning(
                    "Dropping stale Polar group %s task=%s: %s",
                    completed.group_id,
                    completed.task_id,
                    reason,
                )
                continue

            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=staleness,
                policy_version=completed.policy_version,
                scheduler_group_id=completed.group_id,
            )
            accepted.append(completed)

        if accepted:
            self._mark_delivered(len(accepted))
        with self._state_lock:
            self._completed_buffer_size = len(self._completed_buffer)
        return accepted

    def queue_size(self) -> int:
        with self._state_lock:
            return (
                self.output_queue.qsize()
                + self._completed_buffer_size
                + self.deferred_queue.qsize()
            )

    def snapshot_metrics(self) -> dict[str, float]:
        with self._state_lock:
            out = dict(self._metrics)
            out["polar/scheduler/active_groups"] = float(self._active_groups)
            out["polar/scheduler/active_sessions"] = float(self._active_sessions)
            out["polar/scheduler/completed_buffer"] = float(self._completed_buffer_size)
            out["polar/scheduler/output_queue"] = float(self.output_queue.qsize())
            out["polar/scheduler/deferred_queue"] = float(self.deferred_queue.qsize())
            out["polar/scheduler/requested_groups"] = float(self._requested_groups)
            return out

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        active: dict[Any, _PendingGroup] = {}
        active_session_cost = 0
        wakeup = asyncio.Event()

        callback_server, callback_task = await self._start_callback_listener()
        httpx = _load_httpx()
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                while self._running:
                    done = [t for t in active if t.done()]
                    for t in done:
                        pending = active.pop(t)
                        active_session_cost -= pending.session_cost
                        try:
                            t.result()
                        except Exception as exc:
                            logger.exception("Polar async task failed")
                            self._set_fatal(exc)
                            self._running = False
                    self._record_active_counts(active, active_session_cost)

                    while self._running and self._can_admit_group(active, active_session_cost):
                        try:
                            next_group = self._next_group_for_submission()
                        except Exception as exc:
                            self._set_fatal(exc)
                            self._running = False
                            break
                        if next_group is None:
                            break
                        session_cost = len(next_group.group)
                        if session_cost > self.config.max_session_concurrency:
                            self._set_fatal(
                                PolarRolloutSchedulerError(
                                    f"Prompt group needs {session_cost} sessions but "
                                    f"derived max_session_concurrency is "
                                    f"{self.config.max_session_concurrency}"
                                )
                            )
                            self._running = False
                            break
                        if active_session_cost + session_cost > self.config.max_session_concurrency:
                            self.deferred_queue.put(next_group)
                            break

                        gid = self._group_counter
                        self._group_counter += 1
                        submitted_rollout_id, policy_version = self._rollout_context()
                        pending = _PendingGroup(
                            group_id=gid,
                            group=next_group.group,
                            submitted_rollout_id=submitted_rollout_id,
                            policy_version=policy_version,
                            session_cost=session_cost,
                        )
                        task = asyncio.create_task(
                            self._submit_and_collect(client, pending),
                            name=f"polar-rollout-task-{gid}",
                        )
                        task.add_done_callback(lambda _: wakeup.set())
                        active[task] = pending
                        active_session_cost += session_cost
                        self._record_active_counts(active, active_session_cost)

                    if self._running:
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=0.5)
                        except TimeoutError:
                            pass
                        wakeup.clear()

            if active:
                logger.info("Waiting for %d in-flight Polar tasks", len(active))
                await asyncio.gather(*active.keys(), return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar rollout worker stopped")

    async def _start_callback_listener(self) -> tuple[Any, Any]:
        """Bind a FastAPI listener for TaskResult callbacks."""
        from fastapi import FastAPI
        import uvicorn

        app = FastAPI()

        @app.post("/callback/task_result")
        async def on_task_result(request: _FastAPIRequest) -> dict[str, Any]:
            payload = await request.json()
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not task_id:
                return {"ok": False, "reason": "missing task_id"}
            try:
                result = _load_task_result_type().model_validate(payload)
            except Exception:
                logger.exception("Invalid callback payload for task %s", task_id)
                return {"ok": False, "reason": "invalid payload"}
            self._task_results[task_id] = result
            event = self._task_events.get(task_id)
            if event is not None:
                event.set()
            return {"ok": True}

        config = uvicorn.Config(
            app=app, host=self.config.callback_host, port=0,
            log_level="warning", lifespan="on",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name="polar-callback-listener")
        deadline = time.monotonic() + 10.0
        while not server.started:
            if task.done():
                exc = task.exception()
                raise RuntimeError(
                    f"Polar callback listener failed to bind "
                    f"{self.config.callback_host}:0: {exc}"
                ) from (exc if isinstance(exc, BaseException) else None)
            if time.monotonic() > deadline:
                server.should_exit = True
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except TimeoutError:
                    logger.warning("Polar callback listener did not stop within 5s")
                raise RuntimeError(
                    f"Polar callback listener timed out binding "
                    f"{self.config.callback_host}:0"
                )
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self._callback_url = f"http://{self.config.callback_host}:{port}/callback/task_result"
        logger.info("Polar trainer callback listener bound to %s", self._callback_url)
        return server, task

    async def _submit_and_collect(self, client: Any, pending: _PendingGroup) -> None:
        last_error: BaseException | None = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            if not self._running:
                break
            try:
                completed = await self._submit_attempt(client, pending)
                await self._emit_completed(completed)
                return
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts and _is_retriable_polar_task_error(exc):
                    logger.warning(
                        "Polar group %s attempt %d/%d failed (%s); retrying submit",
                        pending.group_id,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    await asyncio.sleep(min(2.0 * attempt, 5.0))
                    continue
                break

        if last_error is None:
            return

        if _is_zero_trainable_error(last_error):
            category_metric = "polar/dropped_zero_trainable_groups"
            reason = "zero trainable tokens"
        elif isinstance(last_error, PolarLowCompleteAcceptFractionError):
            category_metric = "polar/dropped_low_complete_fraction_groups"
            reason = "low complete accept fraction"
        elif isinstance(last_error, RolloutLogprobError):
            category_metric = "polar/dropped_logprob_error_groups"
            reason = "rollout logprob error"
        else:
            category_metric = "polar/dropped_failed_groups"
            reason = "task failure"

        self._inc_metric("polar/dropped_groups")
        self._inc_metric(category_metric)
        self._inc_metric("polar/dropped_sessions", pending.session_cost)
        logger.warning(
            "Dropping Polar group %s because of %s: %s",
            pending.group_id,
            reason,
            last_error,
        )
        return

    async def _submit_attempt(
        self,
        client: Any,
        pending: _PendingGroup,
    ) -> _CompletedGroup:
        payload = _build_task_payload(
            args=self.args, config=self.config, group=pending.group,
            rollout_id=pending.group_id, task_position=0,
        )
        payload["task_id"] = str(payload["task_id"])
        _attach_scheduler_metadata(
            payload,
            group_id=pending.group_id,
            policy_version=pending.policy_version,
            rollout_step=pending.submitted_rollout_id,
        )
        task_result = await self._submit_with_callback(client, payload)

        rejection_reason = self._task_rejection_reason(task_result, pending.group)
        if rejection_reason is not None:
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            )

        group_samples = _convert_task_result_to_samples(
            self.config, task_result, pending.group,
            max_tokens=_resolve_max_tokens(self.args),
        )
        if not group_samples:
            raise PolarRolloutSchedulerError(f"Task {task_result.task_id} converted to zero samples")
        if not _has_trainable_tokens(group_samples):
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} produced zero trainable tokens"
            )
        rejection_reason = _low_complete_accept_fraction_rejection_reason(
            self.config, task_result, group_samples
        )
        if rejection_reason is not None:
            raise PolarLowCompleteAcceptFractionError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            )

        return _CompletedGroup(
            group_id=pending.group_id,
            group=pending.group,
            samples=group_samples,
            task_id=task_result.task_id,
            submitted_rollout_id=pending.submitted_rollout_id,
            policy_version=pending.policy_version,
            session_count=len(task_result.results),
        )

    async def _emit_completed(self, completed: _CompletedGroup) -> None:
        while self._running:
            try:
                self.output_queue.put_nowait(completed)
                self._inc_metric("polar/completed_groups")
                return
            except queue.Full:
                self._inc_metric("polar/output_queue_full_waits")
                await asyncio.sleep(0.1)

    def _next_group_for_submission(self) -> _DeferredGroup | None:
        try:
            deferred = self.deferred_queue.get_nowait()
            self._inc_metric("polar/deferred_queue_dequeues")
            return deferred
        except queue.Empty:
            pass

        groups = self.data_source.get_samples(1)
        if not groups:
            return None
        group = groups[0]
        if not group:
            raise PolarRolloutSchedulerError("Miles data source returned an empty sample group")
        return _DeferredGroup(group=group)

    def _can_admit_group(
        self,
        active: dict[Any, _PendingGroup],
        active_session_cost: int,
    ) -> bool:
        requested_groups = self._shared_requested_groups()
        if requested_groups <= 0:
            return False
        if len(active) >= self.config.max_concurrency:
            return False
        if active_session_cost >= self.config.max_session_concurrency:
            return False
        owned_groups = (
            len(active)
            + self.output_queue.qsize()
            + self._shared_completed_buffer_size()
            + self.deferred_queue.qsize()
        )
        admission_window = min(
            requested_groups,
            self._batch_size * self.config.max_async_level,
        )
        return owned_groups < admission_window

    def _task_rejection_reason(self, task_result: TaskResult, group: list[Any]) -> str | None:
        if not task_result.results:
            return "empty task results"
        if len(task_result.results) != len(group):
            return f"session count {len(task_result.results)} != expected {len(group)}"
        return None

    def _rollout_context(self) -> tuple[int, int]:
        with self._state_lock:
            return self._current_rollout_id, self._current_rollout_id

    def _shared_requested_groups(self) -> int:
        with self._state_lock:
            return self._requested_groups

    def _mark_delivered(self, count: int) -> None:
        with self._state_lock:
            self._requested_groups = max(0, self._requested_groups - int(count))

    def _shared_completed_buffer_size(self) -> int:
        with self._state_lock:
            return self._completed_buffer_size

    def _record_active_counts(
        self,
        active: dict[Any, _PendingGroup],
        active_session_cost: int,
    ) -> None:
        with self._state_lock:
            self._active_groups = len(active)
            self._active_sessions = active_session_cost

    def _inc_metric(self, key: str, amount: float = 1.0) -> None:
        with self._state_lock:
            self._metrics[key] = self._metrics.get(key, 0.0) + amount

    def _set_fatal(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = exc

    async def _submit_with_callback(
        self, client: Any, payload: dict[str, Any]
    ) -> Any:
        """Submit a task, wait on its completion event, and fall back to polling."""
        task_id = payload["task_id"]
        # Register event BEFORE submit so a fast callback cannot arrive first.
        event = asyncio.Event()
        self._task_events[task_id] = event
        payload["callback_url"] = self._callback_url
        base_url = self.config.rollout_server_url
        try:
            resp = await client.post(
                f"{base_url}/rollout/task/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return await self._await_task_result(
                client, task_id, event, task_timeout=self.config.request_timeout
            )
        finally:
            self._task_events.pop(task_id, None)
            self._task_results.pop(task_id, None)

    async def _await_task_result(
        self,
        client: Any,
        task_id: str,
        event: asyncio.Event,
        *,
        task_timeout: float | None,
    ) -> Any:
        """Wait for callback/poll completion with a bounded total task wait."""
        base_url = self.config.rollout_server_url
        deadline = None if task_timeout is None else time.monotonic() + task_timeout
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Polar task {task_id} did not reach a terminal state within "
                    f"{task_timeout:g}s"
                )
            try:
                await asyncio.wait_for(event.wait(), timeout=_CALLBACK_FALLBACK_POLL_SECONDS)
            except TimeoutError:
                status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
                status_resp.raise_for_status()
                status = _load_task_status_type().model_validate(status_resp.json())
                if status.status in ("completed", "failed"):
                    return _load_task_result_type()(
                        task_id=task_id, status=status.status,
                        results=status.results, result_paths=status.result_paths,
                    )
                continue
            result = self._task_results.get(task_id)
            if result is not None:
                return result
            # Race: event set but result missing. Poll until Polar reports a
            # terminal state rather than constructing a TaskResult from a
            # nonterminal status.
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
            status = _load_task_status_type().model_validate(status_resp.json())
            if status.status in ("completed", "failed"):
                return _load_task_result_type()(
                    task_id=task_id, status=status.status,
                    results=status.results, result_paths=status.result_paths,
                )


# ---------------------------------------------------------------------------
# One-shot eval rollout
# ---------------------------------------------------------------------------
async def _run_eval_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    config = resolve_polar_slime_config(args)
    eval_datasets = list(getattr(args, "eval_datasets", []) or [])
    if eval_datasets:
        data: dict[str, dict[str, Any]] = {}
        metrics: dict[str, Any] = {}
        for dataset_cfg in eval_datasets:
            dataset_name, dataset_data, dataset_metrics = await _run_eval_dataset(
                args=args,
                config=config,
                rollout_id=rollout_id,
                dataset_cfg=dataset_cfg,
            )
            data[dataset_name] = dataset_data
            metrics.update(_prefix_eval_metrics(dataset_name, dataset_metrics))

        RolloutFnEvalOutput = _load_rollout_eval_output_type()
        return RolloutFnEvalOutput(data=data, metrics=metrics)

    logger.warning(
        "Polar eval called without args.eval_datasets; falling back to the training data source. "
        "Pass --eval-prompt-data to evaluate validation prompts."
    )
    sample_groups = _pull_sample_groups(data_source, args.rollout_batch_size)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=config.eval_dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    RolloutFnEvalOutput = _load_rollout_eval_output_type()
    return RolloutFnEvalOutput(
        data={config.eval_dataset_name: dataset_data},
        metrics=metrics,
    )


async def _run_eval_dataset(
    *,
    args: Any,
    config: PolarSlimeConfig,
    rollout_id: int,
    dataset_cfg: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_name = str(getattr(dataset_cfg, "name", "") or config.eval_dataset_name)
    sample_groups = _load_eval_sample_groups(args, dataset_cfg)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    return dataset_name, dataset_data, metrics


async def _submit_eval_groups(
    *,
    args: Any,
    config: PolarSlimeConfig,
    dataset_name: str,
    rollout_id: int,
    sample_groups: list[list[Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not sample_groups:
        raise ValueError("Polar eval dataset produced no sample groups")

    httpx = _load_httpx()
    timeout = None if config.request_timeout is None else httpx.Timeout(config.request_timeout)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_one(position: int, group: list[Any]) -> Any:
        async with semaphore:
            payload = _build_task_payload(
                args=args, config=config, group=group,
                rollout_id=rollout_id, task_position=position,
            )
            payload["task_id"] = _eval_task_id(
                payload["task_id"],
                dataset_name=dataset_name,
                rollout_id=rollout_id,
                position=position,
            )
            _attach_scheduler_metadata(
                payload,
                group_id=position,
                policy_version=rollout_id,
                rollout_step=rollout_id,
            )
            return await _submit_and_wait_for_task(
                client,
                config.rollout_server_url,
                payload,
                task_timeout=config.request_timeout,
            )

    async with httpx.AsyncClient(timeout=timeout) as client:
        task_results = await asyncio.gather(
            *(_run_one(pos, g) for pos, g in enumerate(sample_groups))
        )

    output_groups: list[list[Any]] = []
    max_tokens = _resolve_max_tokens(args)
    for group, task_result in zip(sample_groups, task_results, strict=True):
        output_groups.append(
            _convert_task_result_to_samples(
                config, task_result, group,
                max_tokens=max_tokens,
            )
        )

    metrics = _build_metrics(
        config,
        task_results,
        output_groups,
        reward_filter="completed",
    )
    flat_samples = [sample for group in output_groups for sample in group]
    reward_samples = _completed_session_samples(flat_samples)

    return {
        "rewards": [_extract_sample_reward(s, config.reward_key) for s in reward_samples],
        "all_rewards": [_extract_sample_reward(s, config.reward_key) for s in flat_samples],
        "truncated": [_is_truncated(s) for s in reward_samples],
        "all_truncated": [_is_truncated(s) for s in flat_samples],
        "samples": flat_samples,
    }, metrics


def _eval_task_id(base_task_id: Any, *, dataset_name: str, rollout_id: int, position: int) -> str:
    """Namespace eval task ids away from train task ids.

    Training ids commonly use ``{rollout_id}-{sample.group_index}``; eval uses
    ``position`` as group index, so eval 11 / item 11 would collide with train
    group 11. A suffix keeps task polling and persisted result dirs separate.
    """
    safe_dataset = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_name
    )
    return f"{base_task_id}-eval-{safe_dataset}-{rollout_id}-{position}"


def _completed_session_samples(samples: list[Any]) -> list[Any]:
    return [
        sample for sample in samples
        if _sample_session_status(sample) == "COMPLETED"
        and not bool(
            (getattr(sample, "metadata", {}) or {})
            .get("polar", {})
            .get("placeholder")
        )
    ]


def _sample_session_status(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    status = polar_meta.get("session_status")
    return getattr(status, "value", status)


def _load_eval_sample_groups(args: Any, dataset_cfg: Any) -> list[list[Any]]:
    Sample = _load_sample_type()
    path = str(dataset_cfg.path)
    input_key = getattr(dataset_cfg, "input_key", None) or getattr(args, "input_key", "prompt")
    label_key = getattr(dataset_cfg, "label_key", None) or getattr(args, "label_key", None)
    metadata_key = getattr(dataset_cfg, "metadata_key", None) or getattr(args, "metadata_key", "metadata")
    tool_key = getattr(dataset_cfg, "tool_key", None) or getattr(args, "tool_key", None)
    group_size = int(
        getattr(dataset_cfg, "n_samples_per_eval_prompt", None)
        or getattr(args, "n_samples_per_eval_prompt", None)
        or 1
    )
    if group_size <= 0:
        raise ValueError("n_samples_per_eval_prompt must be positive")

    groups: list[list[Any]] = []
    sample_index = 0
    for prompt_index, row in enumerate(_read_jsonl_rows(path)):
        if input_key not in row:
            raise KeyError(f"Eval row {prompt_index} in {path} missing input key {input_key!r}")

        metadata = _inject_eval_metadata(dataset_cfg, row.get(metadata_key))
        if tool_key and tool_key in row:
            tools = row[tool_key]
            if isinstance(tools, str):
                tools = json.loads(tools)
            metadata["tools"] = tools

        group: list[Any] = []
        for _ in range(group_size):
            sample = Sample(
                prompt=copy.deepcopy(row[input_key]),
                label=row.get(label_key) if label_key else None,
                metadata=copy.deepcopy(metadata),
                group_index=prompt_index,
                index=sample_index,
            )
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            group.append(sample)
            sample_index += 1
        groups.append(group)

    return groups


def _read_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Eval row {line_number} in {path} is not a JSON object")
            rows.append(row)
    return rows


def _inject_eval_metadata(dataset_cfg: Any, sample_metadata: Any) -> dict[str, Any]:
    inject = getattr(dataset_cfg, "inject_metadata", None)
    if callable(inject):
        metadata = inject(sample_metadata)
    elif isinstance(sample_metadata, dict):
        metadata = dict(sample_metadata)
    else:
        metadata = {}
    return metadata


def _prefix_eval_metrics(dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    prefixed: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("polar/"):
            prefixed[f"polar/eval/{dataset_name}/{key.removeprefix('polar/')}"] = value
        else:
            prefixed[f"polar/eval/{dataset_name}/{key}"] = value
    return prefixed


def _pull_sample_groups(data_source: Any, batch_size: int) -> list[list[Any]]:
    getter = getattr(data_source, "get_samples", None)
    if callable(getter):
        groups = getter(batch_size)
    elif callable(data_source):
        groups = data_source(batch_size)
    else:
        raise ValueError("data_source must expose get_samples(num_samples) or be callable")
    if not isinstance(groups, list):
        raise ValueError("data_source.get_samples must return a list of sample groups")
    for group in groups:
        if not group:
            raise ValueError("Miles data source returned an empty sample group")
    return groups


def _build_metrics(
    config: PolarSlimeConfig,
    task_results: list[TaskResult],
    output_groups: list[list[Any]],
    *,
    reward_filter: str = "all",
) -> dict[str, Any]:
    flat_samples = [sample for group in output_groups for sample in group]
    all_rewards = [_extract_sample_reward(s, config.reward_key) for s in flat_samples]
    completed_rewards = [
        _extract_sample_reward(s, config.reward_key)
        for s in _completed_session_samples(flat_samples)
    ]
    if reward_filter == "all":
        rewards = all_rewards
    elif reward_filter == "completed":
        rewards = completed_rewards
    else:
        raise ValueError("reward_filter must be 'all' or 'completed'")
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(flat_samples, rewards, config.reward_key))
    return metrics


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_rollout_polar_async(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Miles-compatible async rollout entrypoint.

    Training runs are served by a persistent background worker that pulls
    from ``data_source`` and drains completed groups on each call.
    Evaluation runs are served by a one-shot submit+poll batch over the
    same async HTTP surface.
    """
    if evaluation:
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))

    async_worker = get_global_async_worker(args, data_source)
    async_worker.set_rollout_context(rollout_id)
    target = getattr(args, "rollout_batch_size", 1)
    async_worker.request_groups(int(target))

    data: list[list[Any]] = []
    start = time.monotonic()
    last_progress = start

    while len(data) < target:
        made_progress = False
        completed_groups = async_worker.drain_completed(
            max_groups=target - len(data),
            rollout_id=rollout_id,
        )
        for completed in completed_groups:
            data.append(completed.samples)
            made_progress = True

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, accepted=%d/%d",
                async_worker.queue_size(), len(data), target,
            )
            last_progress = now

        if len(data) < target:
            time.sleep(0.05)

    elapsed = time.monotonic() - start
    logger.info("Async rollout collected %d groups in %.1fs (queue=%d)", len(data), elapsed, async_worker.queue_size())

    _dump_all_trajectories(rollout_id, data)
    _maybe_dump_longest_trace_artifact(rollout_id, data)

    RolloutFnTrainOutput = _load_rollout_train_output_type()
    flat = [s for g in data for s in g]
    rewards = [_extract_sample_reward(s, async_worker.config.reward_key) for s in flat]
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(flat, rewards, async_worker.config.reward_key))
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def _dump_all_trajectories(rollout_id: int, data: list[list[Any]]) -> Path | None:
    """Persist every training trace as a compressed JSONL record."""
    configured_dir = os.environ.get("MILES_POLAR_TRAJECTORY_DUMP_DIR", "").strip()
    if not configured_dir:
        return None

    dump_dir = Path(configured_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    final_path = dump_dir / f"rollout_{int(rollout_id):06d}.jsonl.gz"
    temp_path = dump_dir / f".{final_path.name}.tmp"
    trace_count = 0

    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as stream:
            for group_position, group in enumerate(data):
                for sample_position, sample in enumerate(group):
                    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar") or {}
                    trace_debug = polar_meta.get("trace_debug") or {}
                    status = getattr(sample, "status", None)
                    loss_mask = list(getattr(sample, "loss_mask", None) or [])
                    record = {
                        "rollout_id": int(rollout_id),
                        "group_position": group_position,
                        "sample_position": sample_position,
                        "group_index": getattr(sample, "group_index", None),
                        "sample_index": getattr(sample, "index", None),
                        "session_id": polar_meta.get("session_id"),
                        "task_id": polar_meta.get("task_id"),
                        "node_id": polar_meta.get("node_id"),
                        "trace_index": polar_meta.get("trace_index"),
                        "status": getattr(status, "value", status),
                        "finish_reason": trace_debug.get("finish_reason"),
                        "reward": getattr(sample, "reward", None),
                        "response_length": int(getattr(sample, "response_length", 0) or 0),
                        "prompt_messages": sample.prompt if isinstance(getattr(sample, "prompt", None), list) else [],
                        "response_messages": trace_debug.get("response_messages") or [],
                        "tokens": list(getattr(sample, "tokens", None) or []),
                        "loss_mask": loss_mask,
                        "rollout_log_probs": list(getattr(sample, "rollout_log_probs", None) or []),
                        "trainable_token_count": sum(1 for value in loss_mask if value),
                        "polar_metadata": {
                            key: value for key, value in polar_meta.items() if key != "trace_debug"
                        },
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, default=str))
                    stream.write("\n")
                    trace_count += 1
        temp_path.replace(final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        logger.exception("Failed to persist trajectory dump for rollout=%d", rollout_id)
        return None

    logger.info(
        "Persisted trajectory dump rollout=%d traces=%d path=%s",
        rollout_id,
        trace_count,
        final_path,
    )
    return final_path


def _maybe_dump_longest_trace_artifact(
    rollout_id: int, data: list[list[Any]], *, interval: int = _LONGEST_TRACE_ARTIFACT_INTERVAL
) -> None:
    """Dump the longest session in this rollout's batch as a wandb artifact.

    Groups samples by ``session_id``, picks the session with the largest
    aggregated assistant tokens, and writes its full message chain (per
    trace) to a JSON artifact. Silently no-ops if wandb isn't initialized.
    """
    if interval <= 0 or rollout_id % interval != 0:
        return
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return

    by_session: dict[str, list[Any]] = {}
    for group in data:
        for sample in group:
            sid = _sample_session_id(sample) or "unknown"
            by_session.setdefault(sid, []).append(sample)
    if not by_session:
        return

    def _session_tokens(samples: list[Any]) -> int:
        return sum(int(getattr(s, "response_length", 0) or 0) for s in samples)

    longest_sid, longest_samples = max(by_session.items(), key=lambda kv: _session_tokens(kv[1]))
    total_tokens = _session_tokens(longest_samples)
    if total_tokens <= 0:
        return

    longest_samples = sorted(
        longest_samples,
        key=lambda s: int((s.metadata.get("polar") or {}).get("trace_index", 0) or 0),
    )
    traces = []
    for sample in longest_samples:
        polar_meta = sample.metadata.get("polar") or {}
        trace_debug = polar_meta.get("trace_debug") or {}
        status = getattr(sample, "status", None)
        traces.append({
            "trace_index": polar_meta.get("trace_index"),
            "finish_reason": trace_debug.get("finish_reason"),
            "response_length": int(getattr(sample, "response_length", 0) or 0),
            "status": getattr(status, "value", None) if status is not None else None,
            "prompt_messages": sample.prompt if isinstance(sample.prompt, list) else [],
            "response_messages": trace_debug.get("response_messages") or [],
        })

    first = longest_samples[0]
    first_meta = first.metadata.get("polar") or {}
    reward = getattr(first, "reward", None)
    if isinstance(reward, dict):
        session_reward = float(reward.get("score", 0.0))
    elif isinstance(reward, (int, float)):
        session_reward = float(reward)
    else:
        session_reward = 0.0

    payload = {
        "rollout_id": int(rollout_id),
        "session_id": longest_sid,
        "task_id": first_meta.get("task_id"),
        "node_id": first_meta.get("node_id"),
        "total_assistant_tokens": int(total_tokens),
        "session_reward": session_reward,
        "num_traces": len(traces),
        "traces": traces,
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / f"longest_trace_r{rollout_id}.json"
            fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            artifact = wandb.Artifact(
                name=f"longest_trace_r{rollout_id}", type="rollout-trace"
            )
            artifact.add_file(str(fpath))
            wandb.run.log_artifact(artifact)
    except Exception:
        logger.exception("Failed to log longest-trace wandb artifact")
        return

    logger.info(
        "Logged longest-trace artifact rollout=%d session=%s traces=%d tokens=%d",
        rollout_id, longest_sid, len(traces), total_tokens,
    )


def _group_index_for(group: list[Any]) -> int:
    if group and getattr(group[0], "group_index", None) is not None:
        return int(group[0].group_index)
    return -1


def _extract_sample_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0


def _polar_extra_metrics(
    flat_samples: list[Any],
    rewards: list[float],
    reward_key: str,
) -> dict[str, float]:
    """Compact user-facing Polar metrics for W&B."""
    out: dict[str, float] = {}
    seen: set[str] = set()
    register_to_init_queue_ms: list[float] = []
    init_ms: list[float] = []
    run_ms: list[float] = []
    postrun_ms: list[float] = []
    session_is_placeholder: dict[str, bool] = {}
    session_report: dict[str, dict[str, Any]] = {}
    completed_session_rewards: list[float] = []
    policy_staleness: list[float] = []
    efficiency_weights: list[float] = []
    efficiency_bonuses: list[float] = []
    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        efficiency = polar_meta.get("group_relative_efficiency") or {}
        if isinstance(efficiency, dict):
            if efficiency.get("enabled"):
                efficiency_weights.append(float(efficiency.get("weight", 0.0)))
            efficiency_bonuses.append(float(efficiency.get("bonus", 0.0)))
        if "policy_staleness" in polar_meta:
            policy_staleness.append(float(polar_meta["policy_staleness"]))
        session_id = polar_meta.get("session_id")
        is_placeholder = bool(polar_meta.get("placeholder"))
        if not session_id:
            continue
        if session_id not in seen:
            seen.add(session_id)
            timing = polar_meta.get("timing") or {}
            if timing:
                register_to_init_queue_ms.append(
                    float(timing.get("register_to_init_queue_ms", 0.0))
                )
                init_ms.append(float(timing.get("init_ms", 0.0)))
                run_ms.append(float(timing.get("run_ms", 0.0)))
                postrun_ms.append(float(timing.get("postrun_ms", 0.0)))
            session_is_placeholder[session_id] = is_placeholder
            evaluation = (polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}
            report = evaluation.get("report") or {}
            if isinstance(report, dict) and report:
                session_report[session_id] = report
            if _sample_session_status(sample) == "COMPLETED" and not is_placeholder:
                completed_session_rewards.append(
                    _extract_sample_reward(sample, reward_key)
                )

    if init_ms:
        out["polar/session_ms/register_to_init_queue_mean"] = (
            sum(register_to_init_queue_ms) / len(register_to_init_queue_ms)
        )
        out["polar/session_ms/init_mean"] = sum(init_ms) / len(init_ms)
        out["polar/session_ms/run_mean"] = sum(run_ms) / len(run_ms)
        out["polar/session_ms/postrun_mean"] = sum(postrun_ms) / len(postrun_ms)
    if rewards:
        out["polar/reward_mean"] = sum(rewards) / len(rewards)
    if len(rewards) > 1:
        out["polar/reward_std"] = statistics.pstdev(rewards)
    if completed_session_rewards:
        out["polar/reward_mean_completed"] = (
            sum(completed_session_rewards) / len(completed_session_rewards)
        )
    if policy_staleness:
        out["polar/staleness/mean"] = sum(policy_staleness) / len(policy_staleness)
    if efficiency_weights:
        out["polar/group_efficiency/enabled"] = 1.0
        out["polar/group_efficiency/weight"] = sum(efficiency_weights) / len(efficiency_weights)
    elif efficiency_bonuses:
        out["polar/group_efficiency/enabled"] = 0.0
    if efficiency_bonuses:
        out["polar/group_efficiency/bonus_mean"] = sum(efficiency_bonuses) / len(efficiency_bonuses)

    total_sessions = len(seen)
    empty_sessions = sum(1 for p in session_is_placeholder.values() if p)
    if total_sessions > 0:
        out["polar/rollout_success_rate"] = (
            total_sessions - empty_sessions
        ) / total_sessions
    if session_report:
        graded_sessions = len(session_report)
        resolved = sum(1 for r in session_report.values() if r.get("resolved"))
        out["polar/eval/resolved_rate"] = resolved / graded_sessions
    return out


def _is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == "truncated"


def _load_rollout_train_output_type() -> Any:
    try:
        from miles.rollout.base_types import RolloutFnTrainOutput
    except ImportError as exc:
        raise ImportError(
            "Miles is required to run Polar rollouts from a Miles trainer."
        ) from exc
    return RolloutFnTrainOutput


def _load_rollout_eval_output_type() -> Any:
    try:
        from miles.rollout.base_types import RolloutFnEvalOutput
    except ImportError as exc:
        raise ImportError(
            "Miles is required to run Polar evaluation rollouts from a Miles trainer."
        ) from exc
    return RolloutFnEvalOutput


def _load_sample_type() -> Any:
    try:
        from miles.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Miles is required to build Polar evaluation samples from eval datasets."
        ) from exc
    return Sample


def _flatten_content(content: Any) -> str:
    """Render OpenAI-style message content into a plain string.

    Accepts the three shapes the Chat Completions API uses: ``None`` → ``""``,
    ``str`` returned as-is, ``list[dict]`` with each dict's ``"text"`` field
    concatenated. Anything else is coerced via ``str(content)``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _prompt_to_instruction_text(prompt: Any) -> str:
    """Flatten a dataset prompt (str or chat-message list) into instruction text.

    Single-role lists (e.g. just ``[{"role": "user", "content": ...}]``, which
    is how we shape prompts for VLM checkpoints that require list form) render
    as the bare content so the instruction template sees the same text as the
    string-prompt path. Multi-role lists fall back to ``[role] content`` blocks
    joined by blank lines for a symmetric view of conversation data.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        messages = [m for m in prompt if isinstance(m, dict)]
        contents = [_flatten_content(m.get("content")) for m in messages]
        roles = {str(m.get("role", "user")) for m in messages}
        if len(roles) <= 1:
            return "\n\n".join(c for c in contents if c)
        parts: list[str] = []
        for message, content in zip(messages, contents, strict=True):
            if content:
                role = str(message.get("role", "user"))
                parts.append(f"[{role}] {content}")
        return "\n\n".join(parts)
    if prompt is None:
        return ""
    return str(prompt)


atexit.register(stop_global_worker)

__all__ = [
    "PolarRolloutSchedulerError",
    "PolarLowCompleteAcceptFractionError",
    "AsyncPolarRolloutWorker",
    "get_global_async_worker",
    "stop_global_worker",
    "generate_rollout_polar_async",
    "_DeferredGroup",
    "_PendingGroup",
    "_CompletedGroup",
    "_POLL_INTERVAL",
    "_CALLBACK_FALLBACK_POLL_SECONDS",
    "_LONGEST_TRACE_ARTIFACT_INTERVAL",
    "_build_task_payload",
    "_attach_scheduler_metadata",
    "_submit_and_wait_for_task",
    "_resolve_max_tokens",
    "_convert_task_result_to_samples",
    "_trainable_token_count",
    "_has_trainable_tokens",
    "_low_complete_accept_fraction_rejection_reason",
    "_completed_trainable_session_count",
    "_sample_session_id",
    "_is_zero_trainable_error",
    "_annotate_accepted_samples",
    "_run_eval_rollout",
    "_run_eval_dataset",
    "_submit_eval_groups",
    "_eval_task_id",
    "_completed_session_samples",
    "_sample_session_status",
    "_load_eval_sample_groups",
    "_read_jsonl_rows",
    "_inject_eval_metadata",
    "_prefix_eval_metrics",
    "_pull_sample_groups",
    "_build_metrics",
    "_maybe_dump_longest_trace_artifact",
    "_group_index_for",
    "_extract_sample_reward",
    "_polar_extra_metrics",
    "_is_truncated",
    "_load_rollout_train_output_type",
    "_load_rollout_eval_output_type",
    "_load_sample_type",
    "_load_task_result_type",
    "_load_task_status_type",
    "_flatten_content",
    "_prompt_to_instruction_text",
]
