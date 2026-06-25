"""Supervisor / lifecycle for the multi-process session server (workers > 1).

Spawns N headless workers + 1 thin router under a multiprocessing **spawn**
context, wires one ``socket.socketpair()`` per worker (worker-end to the worker,
router-end to the router), waits until ALL workers are ready (the router's
``/health`` reports every worker healthy), and then monitors the children:

* fail-fast — on ANY child death the whole group is killed and the failure is
  surfaced to the caller (a background thread cannot ``raise`` into the main
  rollout path, so it sets a flag the caller polls / the next launch observes);
* no orphans — SIGTERM / ``atexit`` / crash all route through one group kill, and
  every child sets ``PR_SET_PDEATHSIG=SIGKILL`` so a parent crash reaps them too.

The workers=1 path does NOT use this module; it stays a single
``run_session_server`` process with no router / IPC (see ``router_manager``).
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

from miles.rollout.session.session_router import run_router
from miles.rollout.session.session_worker import run_worker
from miles.utils.http_utils import wait_for_server_ready

logger = logging.getLogger(__name__)

# How long the readiness gate waits for all workers' tokenizer/TITO init + the
# router's all-workers-healthy report after the port is accepting connections.
_READINESS_TIMEOUT = 600.0
# Grace period between SIGTERM and SIGKILL when tearing the group down.
_TERM_GRACE = 5.0


class SessionServerSupervisor:
    """Owns the router + worker processes and their lifecycle.

    Lives in the caller's process (the rollout / Ray-actor main path). Holds no
    session state; just the child processes and the fail-fast machinery.
    """

    def __init__(self, args, backend_url: str, ip: str, port: int):
        self.args = args
        self.backend_url = backend_url
        self.ip = ip
        self.port = port
        self.n_worker = int(getattr(args, "session_server_workers", 1))
        if self.n_worker < 1:
            raise ValueError(f"session_server_workers must be >= 1, got {self.n_worker}")

        self._ctx = multiprocessing.get_context("spawn")
        self._workers: list[multiprocessing.process.BaseProcess] = []
        self._router: multiprocessing.process.BaseProcess | None = None
        self._monitor: threading.Thread | None = None
        self._stop = threading.Event()
        self._shutdown_done = threading.Event()
        self._failure: str | None = None

    # ---- launch ---------------------------------------------------------

    def start(self) -> None:
        """Spawn children, wire fds, and block until all workers are ready.

        Raises ``RuntimeError`` if any child dies before / during readiness.
        """
        worker_ends: list[socket.socket] = []
        router_ends: list[socket.socket] = []
        for _ in range(self.n_worker):
            a, b = socket.socketpair()
            worker_ends.append(a)
            router_ends.append(b)

        try:
            for i in range(self.n_worker):
                p = self._ctx.Process(
                    target=run_worker,
                    args=(self.args, self.backend_url, worker_ends[i], i),
                    name=f"miles-session-worker-{i}",
                    daemon=False,  # daemon procs cannot spawn children; also we own teardown
                )
                p.start()
                self._workers.append(p)

            self._router = self._ctx.Process(
                target=run_router,
                args=(self.args, router_ends, self.ip, self.port),
                name="miles-session-router",
                daemon=False,
            )
            self._router.start()
        finally:
            # The parent owns neither end: close every fd here so EOF is
            # observable on peer death (the spawned children received their own
            # dup'd copies). Children close the ends they don't own below.
            for s in worker_ends + router_ends:
                _safe_close(s)

        atexit.register(self.shutdown)
        self._install_signal_handlers()

        try:
            self._await_ready()
        except Exception:
            self.shutdown()
            raise

        self._monitor = threading.Thread(target=self._monitor_loop, name="session-supervisor-monitor", daemon=True)
        self._monitor.start()

    def _await_ready(self) -> None:
        """Block until the router serves and reports all workers healthy."""
        # First the TCP port (router started); then the all-workers-healthy gate.
        wait_for_server_ready(self.ip, self.port, self._router, timeout=_READINESS_TIMEOUT)
        deadline = time.time() + _READINESS_TIMEOUT
        while time.time() < deadline:
            self._raise_if_child_dead()
            if self._health_all_ok():
                logger.info(
                    "[session-supervisor] router ready at %s:%s, all %d workers healthy",
                    self.ip,
                    self.port,
                    self.n_worker,
                )
                return
            time.sleep(0.25)
        raise RuntimeError(
            f"session server not ready after {_READINESS_TIMEOUT}s "
            f"(router {self.ip}:{self.port}, {self.n_worker} workers)"
        )

    def _health_all_ok(self) -> bool:
        import httpx

        try:
            resp = httpx.get(f"http://{self.ip}:{self.port}/health", timeout=10.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200 and resp.json().get("status") == "ok"

    # ---- monitoring / fail-fast -----------------------------------------

    def _children(self):
        procs = list(self._workers)
        if self._router is not None:
            procs.append(self._router)
        return procs

    def _raise_if_child_dead(self) -> None:
        for p in self._children():
            if not p.is_alive():
                raise RuntimeError(
                    f"session server child {p.name} (pid={p.pid}) died with exitcode={p.exitcode} during startup"
                )

    def _monitor_loop(self) -> None:
        """Poll children; on ANY death, record the failure and kill the group.

        A thread ``raise`` does not reach the main rollout path, so the failure
        is recorded in ``self._failure`` (observed via :meth:`check`) AND the
        whole group is torn down so the rollout cannot keep using a half-dead
        server.
        """
        while not self._stop.is_set():
            dead = [p for p in self._children() if not p.is_alive()]
            if dead:
                names = ", ".join(f"{p.name}(pid={p.pid}, exit={p.exitcode})" for p in dead)
                self._failure = f"session server child died: {names}"
                logger.error("[session-supervisor] FAIL-FAST: %s; killing the session-server group", names)
                self.shutdown()
                return
            time.sleep(0.5)

    def check(self) -> None:
        """Raise if the monitor recorded a child death (poll from rollout path)."""
        if self._failure is not None:
            raise RuntimeError(self._failure)

    @property
    def failed(self) -> bool:
        return self._failure is not None

    # ---- teardown -------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        # Best-effort: only the main thread can install handlers. A worker /
        # Ray actor thread launch falls back to atexit + pdeathsig.
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)

                def _handler(signum, frame, _prev=prev):
                    self.shutdown()
                    # Chain to the previous disposition so the SIGNAL still has
                    # its original effect on THIS process: a callable runs; an
                    # ignored signal stays ignored; the default (SIG_DFL) must
                    # actually terminate us — restore SIG_DFL and re-raise, else
                    # a Ray/cluster SIGTERM would reap our children but leave the
                    # rollout process alive.
                    if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                        _prev(signum, frame)
                    elif _prev == signal.SIG_DFL:
                        signal.signal(signum, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)

                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    def shutdown(self) -> None:
        """Terminate router + all workers; SIGKILL after a short grace. Idempotent."""
        if self._shutdown_done.is_set():
            return
        self._shutdown_done.set()
        self._stop.set()

        procs = self._children()
        for p in procs:
            _terminate(p)
        deadline = time.time() + _TERM_GRACE
        for p in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                p.join(timeout=remaining)
            except Exception:
                pass
        for p in procs:
            if p.is_alive():
                _kill(p)
                try:
                    p.join(timeout=2.0)
                except Exception:
                    pass


def _safe_close(s: socket.socket) -> None:
    try:
        s.close()
    except OSError:
        pass


def _terminate(p) -> None:
    try:
        if p.is_alive():
            p.terminate()  # SIGTERM
    except (ValueError, AttributeError, ProcessLookupError):
        pass


def _kill(p) -> None:
    try:
        if p.pid is not None:
            os.kill(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
