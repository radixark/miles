"""Per-episode Daytona sandbox running a HUD environment image.

Lifecycle, and why each piece is here:

- Daytona does not run the image's CMD, and these images serve MCP over stdio
  anyway, so ``start()`` execs the service bring-up itself (Xvfb, the
  persistent context server, and an HTTP re-serve of the stdio MCP server) and
  then polls the control port from outside.
- Concurrency is capped process-wide, independent of the rollout's batch
  geometry, so a wide batch cannot open an unbounded number of cloud sandboxes.
- Deletion is verified rather than fire-and-forget, and a reaper sweeps
  whatever still leaked: an earlier version trusted ``delete()``'s accepted
  request and left a step's worth of sandboxes running every step.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from examples.experimental.hud.mcp_client import SyncMCP

logger = logging.getLogger(__name__)

MCP_PORT = 8765

# One label value per process, so a reaper can distinguish this run's leaks from
# a colleague's live sandboxes in the same Daytona org.
PROC_ID = uuid.uuid4().hex[:8]
LAUNCHER_LABEL = "miles-hud"

_gate: threading.Semaphore | None = None
_gate_lock = threading.Lock()
_reaper_started = False

_START_SCRIPT = r"""
mkdir -p /tmp/hud
cat > /tmp/hud/serve_http.py <<'PYEOF'
import inspect, os
os.environ.setdefault("DISPLAY", ":1")

# The vendored hud fork runs its @mcp.initialize hook for the first MCP session
# only and then awaits None, so any later session dies with -32602. We keep one
# session per sandbox, but make the server survive a reconnect anyway.
from hud.server import low_level as ll

_orig = ll.InitSession.__init__

def _patched(self, *a, init_fn=None, **kw):
    if init_fn is not None:
        inner = init_fn
        async def wrapper(ctx):
            res = inner(ctx)
            if inspect.isawaitable(res):
                await res
        init_fn = wrapper
    _orig(self, *a, init_fn=init_fn, **kw)

ll.InitSession.__init__ = _patched

from hud_controller.server import mcp
mcp.run(transport="http", host="0.0.0.0", port=8765, show_banner=False)
PYEOF
cat > /tmp/hud/start_all.sh <<'SHEOF'
#!/bin/sh
if [ ! -e /tmp/.X11-unix/X1 ]; then Xvfb :1 -screen 0 1920x1080x24 >/dev/null 2>&1 & fi
while [ ! -e /tmp/.X11-unix/X1 ]; do sleep 0.2; done
export DISPLAY=:1
cd /app
python3 -m hud_controller.context >/tmp/hud/ctx.log 2>&1 &
sleep 2
python3 /tmp/hud/serve_http.py >/tmp/hud/mcp.log 2>&1 &
SHEOF
chmod +x /tmp/hud/start_all.sh
nohup setsid /tmp/hud/start_all.sh >/dev/null 2>&1 &
echo started
"""


def resolve_api_key() -> str:
    """Key from the environment, else from a file.

    A path is forwarded rather than a value on multi-host clusters: ray's
    runtime_env records env values in plaintext.
    """
    if key := os.environ.get("DAYTONA_API_KEY"):
        return key
    path = os.environ.get("DAYTONA_API_KEY_FILE", "~/.config/daytona/api_key")
    with open(os.path.expanduser(path)) as f:
        return f.read().strip()


def client():
    # In-function on purpose: daytona is not in the training image's
    # requirements, and keeping it out of module scope is what lets the offline
    # tests import this module (and hud_task_env, which imports it) without it.
    from daytona import Daytona, DaytonaConfig

    return Daytona(DaytonaConfig(api_key=resolve_api_key()))


def get_gate(limit: int) -> threading.Semaphore:
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = threading.Semaphore(limit)
        return _gate


def _reap_once(max_age_min: float) -> int:
    """Delete this process's sandboxes older than *max_age_min*."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_min)
    reaped = 0
    try:
        for sb in client().list():
            labels = sb.labels or {}
            if labels.get("launcher") != LAUNCHER_LABEL or labels.get("proc") != PROC_ID:
                continue
            raw = str(getattr(sb, "created_at", "") or "")
            try:
                created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if created < cutoff:
                try:
                    sb.delete()
                    reaped += 1
                except Exception:  # noqa: BLE001 - next sweep retries
                    pass
    except Exception:  # noqa: BLE001 - never let the reaper kill the run
        logger.warning("hud sandbox reaper sweep failed", exc_info=True)
    if reaped:
        logger.warning("hud sandbox reaper deleted %d leaked sandbox(es)", reaped)
    return reaped


def start_reaper(max_age_min: float, interval_s: float = 300.0) -> None:
    """Start the background sweep once per process."""
    global _reaper_started
    with _gate_lock:
        if _reaper_started:
            return
        _reaper_started = True

    def loop() -> None:
        while True:
            time.sleep(interval_s)
            _reap_once(max_age_min)

    threading.Thread(target=loop, name="hud-sandbox-reaper", daemon=True).start()


class HudSandbox:
    """One sandbox, one MCP session, one episode."""

    def __init__(
        self,
        image: str,
        *,
        cpu: int = 2,
        memory_gb: int = 4,
        disk_gb: int = 10,
        max_age_min: float = 20.0,
    ) -> None:
        self.image = image
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.disk_gb = disk_gb
        self.max_age_min = max_age_min
        self.run_id = uuid.uuid4().hex[:8]
        self._sb = None
        self._gate: threading.Semaphore | None = None

    def start(self, gate: threading.Semaphore, ready_timeout_s: float = 300.0):
        """Create the sandbox, bring services up, return a connected SyncMCP."""
        from daytona import CreateSandboxFromImageParams, Resources  # optional dep; see client()

        gate.acquire()
        self._gate = gate
        t0 = time.time()
        d = client()
        # Daytona's own TTL is the last-resort backstop; the reaper is faster.
        self._sb = d.create(
            CreateSandboxFromImageParams(
                image=self.image,
                resources=Resources(cpu=self.cpu, memory=self.memory_gb, disk=self.disk_gb),
                auto_stop_interval=15,
                auto_delete_interval=30,
                labels={"launcher": LAUNCHER_LABEL, "proc": PROC_ID, "run-id": self.run_id},
            ),
            timeout=600,
        )
        self._sb.process.exec(_START_SCRIPT, timeout=90)
        url = self._sb.create_signed_preview_url(MCP_PORT, expires_in_seconds=86400)
        base = url.url if hasattr(url, "url") else str(url)

        deadline = time.time() + ready_timeout_s
        with httpx.Client(timeout=10) as probe:
            while time.time() < deadline:
                try:
                    probe.get(base.rstrip("/") + "/mcp")
                    break
                except Exception:  # noqa: BLE001 - still booting
                    time.sleep(3)

        mcp = SyncMCP(base)
        mcp.initialize()
        logger.info(
            "[hud %s] sandbox %s ready in %.0fs (image=%s)",
            self.run_id,
            self._sb.id[:8],
            time.time() - t0,
            self.image,
        )
        return mcp

    def delete(self) -> None:
        """Delete the sandbox, confirming it is really gone."""
        sb, self._sb = self._sb, None
        try:
            if sb is not None:
                # wait=True: the fire-and-forget default returns on request
                # acceptance, and a rejected-but-accepted request is exactly how
                # a step's worth of sandboxes used to survive its step.
                try:
                    sb.delete(timeout=60, wait=True)
                except TypeError:  # older SDKs have no wait kwarg
                    sb.delete()
        except Exception:  # noqa: BLE001 - the reaper is the backstop
            logger.warning("[hud %s] sandbox delete failed; reaper will retry", self.run_id)
        finally:
            if self._gate is not None:
                self._gate.release()
                self._gate = None
