"""Ordered control plane for an exclusively owned GPU delta receiver.

The tokenizer sends ordered controls for a Miles-owned update.
The committed (stream, version) guards every directional XOR.
Preparation owns immutable buffers; update sets the pause flag, fences readers, pauses generation in the
requested mode (retracting by default), then mutates on the scheduler thread. Failed or ambiguous updates
that might have partially mutated weights require receiver replacement.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from miles.utils.gpu_delta import profiling

# Request fields every transaction receipt repeats as provenance.
_PROVENANCE = (
    "session_id",
    "manifest_sha256",
    "base_stream_id",
    "base_version",
    "stream_id",
    "target_version",
    "plan_digest",
    "reverse",
)


class PreparationDrainError(RuntimeError):
    """A failed discard may still have CUDA readers of this publication."""


@dataclass
class _Transaction:
    request: dict
    destination: tuple[str | None, int]
    release_future: Future | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    state: str = "PREPARING"
    message: str = ""
    retryable_prepare_error: bool | None = None
    prepared: Any = None
    result: dict = field(default_factory=dict)
    pause_started_ns: int | None = None
    reader_fence_completed_ns: int | None = None
    resumed_ns: int | None = None
    nvtx_pause_id: int | None = None

    def poison_release(self, error: BaseException) -> None:
        self.state = "POISONED"
        self.message = f"GPU-delta resource release failed: {type(error).__name__}: {error}; restart required"


class DeltaSession:
    """One original process following the prepare/apply/resume sequence.

    ``prepare`` owns immutable host inputs, small GPU metadata/workspace and CPU
    plans. ``apply`` allocates large decoded slots after the reader fence, joins
    DE/application streams and releases scratch before resume. Error cleanup
    retains buffers until both streams have drained. ``shadow_store`` is the
    canonical shadow store (name -> ``CanonicalShadow``) every backend's
    ``CanonicalShadows`` view wraps; it outlives ``clear`` so every later
    admission finds the committed version.
    """

    def __init__(self, identity: dict, backend: Any, shadow_store: dict):
        self.identity = identity
        self.backend = backend
        self.plan_digest = backend.plan_digest
        self.shadow_store = shadow_store
        self.committed: tuple[str | None, int] = (None, 0)
        self._update: _Transaction | None = None
        # The closed backend's shared cache directory, and whether its arena owns the removal.
        self._closed_cache = None
        self._cache_owner = False
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-delta")

    def status(self) -> dict:
        with self._lock:
            transaction = self._update
            stream_id, version = self.committed
            receipt = {"identity": self.identity, "committed_stream_id": stream_id, "version": version}
            if transaction is None or transaction.state == "CLEARED":
                receipt["state"] = "IDLE" if self.backend is not None else "CLEARED"
            else:
                receipt["state"] = transaction.state
            if transaction is None:
                return receipt
            if transaction.retryable_prepare_error is not None and transaction.state in {"FAILED", "ABORTED"}:
                receipt["retryable_prepare_error"] = transaction.retryable_prepare_error
            started, resumed = transaction.pause_started_ns, transaction.resumed_ns
            return receipt | {
                "message": transaction.message,
                **{key: transaction.request[key] for key in _PROVENANCE},
                "result": transaction.result,
                # One original process's clock, never subtract across ranks.
                # Open/failed pauses have no qualified completed duration.
                "scheduler_timing": {
                    "clock": "monotonic_ns",
                    "pause_started_ns": started,
                    "reader_fence_completed_ns": transaction.reader_fence_completed_ns,
                    "resumed_ns": resumed,
                    "blocked_s": (resumed - started) / 1e9 if resumed is not None and started is not None else None,
                },
            }

    def prepare(self, request: dict) -> Future:
        with self._lock:
            previous = self._update
            if previous is not None:
                if previous.state not in {"RESUMED", "CLEARED"}:
                    raise ValueError("unsettled delta session requires restart")
                if request["session_id"] == previous.request["session_id"] and not previous.cancelled.is_set():
                    raise ValueError("completed delta session must not be replayed")
            if self.backend is None:
                raise ValueError("cleared delta receiver must be described before its next update")
            base = request["base_stream_id"], request["base_version"]
            target = request["stream_id"], request["target_version"]
            if (base[0] is None) != (base[1] == 0) or base[1] < 0 or not target[0] or target[1] <= base[1]:
                raise ValueError(f"invalid GPU-delta publication endpoints: {base} -> {target}")
            source, destination = (target, base) if request["reverse"] else (base, target)
            if source != self.committed:
                raise ValueError(f"GPU-delta source {source} differs from committed {self.committed}")
            transaction = _Transaction(request=request, destination=destination)
            released = previous.release_future if previous is not None else None
            # Submit first: a rejected submission must not replace the settled record. The lock keeps
            # the worker from publishing this transaction's state before it becomes current.
            completion = self._executor.submit(self._prepare, transaction, released)
            # An abort racing PREPARING must join its late discard before clearing the shared arena.
            transaction.release_future = completion
            self._update = transaction
            return completion

    def _transaction(self, session_id: str) -> _Transaction:
        transaction = self._update
        if transaction is None or transaction.request["session_id"] != session_id:
            raise ValueError("delta operation belongs to a different session")
        return transaction

    @profiling.annotate("gpu_delta.prepare")
    def _prepare(self, transaction: _Transaction, released: Future | None) -> dict:
        try:
            if released is not None:
                released.result()
            try:
                prepared = self.backend.prepare(transaction.request, transaction.cancelled)
            except PreparationDrainError:
                raise
            except Exception as exc:
                from miles.utils.gpu_delta.host_worker import retryable_prepare_errno

                with self._lock:
                    if transaction.state == "PREPARING":
                        transaction.state = "FAILED"
                        transaction.message = f"preparation failed: {type(exc).__name__}: {exc}"
                        transaction.retryable_prepare_error = retryable_prepare_errno(exc) is not None
                    return self.status()
            with self._lock:
                if transaction.state == "PREPARING":
                    transaction.prepared = prepared
                    transaction.state = "PREPARED"
                    return self.status()
            prepared.close()
        except BaseException as exc:
            # This includes a failed previous release, partial-prepare drain, or late discard after abort.
            # Keep the failed future (and its resources) until the engine is replaced.
            with self._lock:
                transaction.poison_release(exc)
            raise
        return self.status()

    @profiling.annotate("gpu_delta.scheduler.apply")
    def apply(
        self,
        session_id: str,
        fence: Callable[[], None],
        pause_generation: Callable[[], None],
        flush: Callable[[], bool] | None,
        pause_mode: str = "retract",
    ) -> dict:
        """Fence readers, pause generation, optionally flush, then mutate; ``pause_mode`` names the pause's range."""
        with self._lock:
            transaction = self._transaction(session_id)
            if transaction.state != "PREPARED":
                raise ValueError("delta apply requires a prepared session")
            # The scheduler has stopped new work. From this point a peer may
            # already be mutating. The tokenizer does not abort after apply dispatch.
            transaction.state = "APPLYING"
            transaction.pause_started_ns = time.monotonic_ns()
            if profiling.ENABLED:
                # A process range spans separate apply/resume RPCs. Thread-local
                # push/pop ranges must not be left open across those calls. It ends
                # only after a successful unpause: poisoned, failed-resume and
                # keep_pause=True sessions leave it open, so it is not a completed
                # duration and may outlive this session.
                transaction.nvtx_pause_id = profiling.nvtx.range_start("gpu_delta.pause")
        if profiling.ENABLED:
            fence = profiling.annotate("gpu_delta.pause.reader_fence")(fence)
            # One stable range per mode, gpu_delta.pause.retract or gpu_delta.pause.in_place.
            pause_generation = profiling.annotate(f"gpu_delta.pause.{pause_mode}")(pause_generation)
            if flush is not None:
                flush = profiling.annotate("gpu_delta.pause.flush_cache")(flush)
        try:
            fence()
            transaction.reader_fence_completed_ns = time.monotonic_ns()
            # Never reclaim KV after a failed reader fence.
            pause_generation()
            if flush is not None and not flush():
                raise ValueError("cache flush failed before delta mutation")
            result = transaction.prepared.apply()
        except Exception as exc:
            with self._lock:
                transaction.state = "POISONED"
                transaction.message = f"update failed after pause: {type(exc).__name__}: {exc}; restart required"
                if profiling.ENABLED:
                    # The scheduler is still paused: do not fabricate an end.
                    profiling.nvtx.mark("gpu_delta.pause.poisoned")
            raise
        with self._lock:
            transaction.result = result
            transaction.state = "APPLIED"
            receipt = self.status()
            # Keep the paused IPC acknowledgement small; resume/status retain
            # the full result, including preparation and application metrics.
            receipt["result"] = {
                "applied": result["applied"],
                "target_version": result["target_version"],
            }
            return receipt

    @profiling.annotate("gpu_delta.scheduler.resume")
    def resume(self, session_id: str, resume: Callable[[int], None], keep_pause: bool = False) -> dict:
        with self._lock:
            transaction = self._transaction(session_id)
            if transaction.state != "APPLIED":
                raise ValueError("delta resume requires an applied session")
            transaction.state = "RESUMING"
            try:
                resume(transaction.destination[1])
            except Exception as exc:
                transaction.state = "POISONED"
                transaction.message = f"resume failed: {type(exc).__name__}: {exc}; restart required"
                raise
            self.committed = transaction.destination
            # Version commit does not end a pause retained by the caller.
            if not keep_pause:
                transaction.resumed_ns = time.monotonic_ns()
                if profiling.ENABLED:
                    profiling.nvtx.range_end(transaction.nvtx_pause_id)
                    transaction.nvtx_pause_id = None
            elif profiling.ENABLED:
                profiling.nvtx.mark("gpu_delta.pause.retained")
            transaction.state = "RESUMED"
            prepared, transaction.prepared = transaction.prepared, None
            # Keep host release I/O off the scheduler; this FIFO executor runs
            # it before the next prepare. Resume follows the all-rank apply barrier.
            transaction.release_future = self._executor.submit(prepared.close, True)
            return self.status()

    def abort(self, session_id: str) -> dict:
        with self._lock:
            transaction = self._update
            if transaction is None:
                return self.status()
            # A peer may have rejected admission before this rank accepted the new session.
            if transaction.state in {"CLEARED", "RESUMED"} and (
                transaction.request["session_id"] != session_id or transaction.cancelled.is_set()
            ):
                return self.status()
            transaction = self._transaction(session_id)
            if transaction.state not in {"PREPARING", "PREPARED", "FAILED", "ABORTED"}:
                raise ValueError("cannot abort a delta after apply dispatch")
            transaction.state = "ABORTED"
            # Wake ranks waiting for their shared preparation owner, including
            # when that owner failed before reaching host-cache construction.
            transaction.cancelled.set()
            if transaction.prepared is not None:
                prepared, transaction.prepared = transaction.prepared, None
                transaction.release_future = self._executor.submit(prepared.close)
            return self.status()

    def clear(self) -> dict:
        with self._lock:
            transaction = self._update
            if transaction is not None and transaction.state not in {"RESUMED", "CLEARED", "ABORTED"}:
                raise ValueError("delta state can only be cleared when idle, resumed or aborted before apply")
        # Joining outside the lock lets an aborted prepare finish draining on its worker.
        self._executor.shutdown(wait=True)
        try:
            if transaction is not None and transaction.release_future is not None:
                transaction.release_future.result()
            if self.backend is not None:
                arena = self.backend.host_arena
                self._closed_cache, self._cache_owner = arena.directory, arena.owner
                self.backend.close()
                self.backend = None
        except BaseException as exc:
            if transaction is not None:
                with self._lock:
                    transaction.poison_release(exc)
            raise
        with self._lock:
            if self._update is not None:
                self._update.state = "CLEARED"
                self._update.result = {}
                self._update.release_future = None
            return self.status()

    def release_cache(self) -> dict:
        # The tokenizer sends this only after the all-rank clear barrier. The host arena's
        # preparation owner, one per engine-host, removes the shared cohort directory, never
        # the source publication or another engine's cache.
        if self.backend is not None:
            raise ValueError("delta resources must be closed before cache removal")
        if self._cache_owner:
            # Once its arena closed, another cohort's owner may already have reclaimed this cohort.
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self._closed_cache)
        self._closed_cache, self._cache_owner = None, False
        return self.status()

    def reattach(self, make_backend: Callable[[dict, dict, bool], Any]) -> None:
        """Attach ``make_backend(identity, shadow_store, create)`` once a cleared backend's cache is removed.

        Only a version-0 receiver may create canonical shadows: after an update, only the retained
        shadows hold the committed version.
        """
        if self._closed_cache is not None:
            raise ValueError("delta cache removal has not completed")
        backend = make_backend(self.identity, self.shadow_store, self.committed == (None, 0))
        if backend.plan_digest != self.plan_digest:
            backend.close()
            raise ValueError("GPU-delta checkpoint changed since receiver admission; reload the engine")
        self.backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-delta")
