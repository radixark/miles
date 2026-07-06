# doc-dev: docs/developer/multi-process-session-server.md
"""Supervisor for the multi-process session server — the process-lifecycle layer.

- Runs in the rollout process and is the only way the session server runs; `--session-server-workers=1` (the default) is simply N=1, not a separate chassis.
- `start()` spawns (`"spawn"` context) N workers + 1 router, one `socket.socketpair` per worker, then closes every socket end the parent holds — so a child death is observable as EOF on the surviving peer.
- Readiness: waits until the router's `/health` reports all workers healthy, raising early if a child dies during startup (e.g. while loading its tokenizer) or the deadline elapses; a failed `start()` tears down whatever it already spawned.
- A monitor thread records the first child death and tears the whole group down (fail-fast).
- `check()`, called on the rollout path, re-raises that recorded failure so a dead worker fails the rollout loudly instead of hanging requests routed to its shard.
- `shutdown()` is idempotent and thread-safe (a concurrent caller blocks until teardown completes) — SIGTERM, a short grace period, then SIGKILL — and is also registered via `atexit`, so no orphan processes survive.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
import signal
import socket
import threading
import time

import httpx

from miles.rollout.session.router import run_router
from miles.rollout.session.worker import run_worker
from miles.utils.http_utils import wait_for_server_ready

logger = logging.getLogger(__name__)

_READINESS_TIMEOUT = 120.0
_HEALTH_POLL_INTERVAL = 0.5
_MONITOR_INTERVAL = 0.5
_TERM_GRACE = 5.0


class SessionServerSupervisor:
    """Spawns and supervises N session workers + 1 router; fail-fast on any death."""

    def __init__(self, args, backend_url: str, ip: str, port: int):
        self.args = args
        self.backend_url = backend_url
        self.ip = ip
        self.port = port
        self.n_worker = int(getattr(args, "session_server_workers", 1))
        self._ctx = multiprocessing.get_context("spawn")
        self._procs: list = []
        self._router = None
        self._failure: str | None = None
        self._shutdown_done = False
        self._shutdown_lock = threading.Lock()
        self._monitor: threading.Thread | None = None

    def start(self) -> None:
        worker_ends, router_ends = [], []
        for _ in range(self.n_worker):
            worker_end, router_end = socket.socketpair()
            worker_ends.append(worker_end)
            router_ends.append(router_end)

        # Each child lands in _procs before anything else can fail, so the except
        # path below reaps every process a partially failed start() has spawned.
        try:
            for i in range(self.n_worker):
                p = self._ctx.Process(
                    target=run_worker, args=(self.args, self.backend_url, worker_ends[i], i), daemon=False
                )
                p.start()
                self._procs.append(p)
            router = self._ctx.Process(
                target=run_router, args=(self.args, router_ends, self.ip, self.port), daemon=False
            )
            router.start()
            self._procs.append(router)
            self._router = router

            # Parent closes every socket end so a child death is observable as EOF on its peer.
            for s in worker_ends + router_ends:
                s.close()

            self._await_ready()
        except Exception:
            self.shutdown()
            raise
        self._monitor = threading.Thread(target=self._monitor_loop, name="session-supervisor", daemon=True)
        self._monitor.start()
        atexit.register(self.shutdown)
        logger.info("Session server ready: %d workers + router at %s:%s", self.n_worker, self.ip, self.port)

    def _await_ready(self) -> None:
        # Router opens the TCP port quickly; this also watches the router process liveness.
        wait_for_server_ready(self.ip, self.port, self._router, timeout=_READINESS_TIMEOUT)
        deadline = time.monotonic() + _READINESS_TIMEOUT
        while time.monotonic() < deadline:
            self._raise_if_child_dead()  # a worker may die while loading its tokenizer
            if self._health_ok():
                return
            time.sleep(_HEALTH_POLL_INTERVAL)
        raise RuntimeError(f"session server not healthy within {_READINESS_TIMEOUT}s")

    def _health_ok(self) -> bool:
        try:
            return httpx.get(f"http://{self.ip}:{self.port}/health", timeout=10.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _raise_if_child_dead(self) -> None:
        for p in self._procs:
            if not p.is_alive():
                raise RuntimeError(f"session server child pid={p.pid} died during startup (exitcode={p.exitcode})")

    def _monitor_loop(self) -> None:
        while not self._shutdown_done:
            for p in self._procs:
                if not p.is_alive():
                    # shutdown() flips _shutdown_done before terminate(), so a death seen
                    # while the flag is still False cannot be shutdown-caused.
                    if self._shutdown_done:
                        return
                    self._failure = f"session server child pid={p.pid} died (exitcode={p.exitcode})"
                    logger.error("%s; tearing down session server", self._failure)
                    self.shutdown()
                    return
            time.sleep(_MONITOR_INTERVAL)

    def check(self) -> None:
        """Raise on the rollout path if any session-server child has died (no silent hang)."""
        if self._failure is not None:
            raise RuntimeError(f"session server failed: {self._failure}")

    def shutdown(self) -> None:
        # The lock makes a concurrent caller block until teardown completes instead of
        # returning while children are still being reaped.
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
            for p in self._procs:
                if p.is_alive():
                    p.terminate()
            deadline = time.monotonic() + _TERM_GRACE
            for p in self._procs:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
            for p in self._procs:
                if p.is_alive():
                    try:
                        os.kill(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
