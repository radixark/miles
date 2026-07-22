"""Tests for the Daytona sandbox half: ownership labels and the orphan-TTL
keepalive lifecycle.

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the recipe:

    pytest examples/experimental/openenv/tests/ -q
"""

import inspect
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tb2_sandbox_daytona as sandbox  # noqa: E402


def test_sandbox_labels_default_to_unix_user(monkeypatch):
    monkeypatch.delenv("OPENENV_LAUNCHER", raising=False)
    monkeypatch.delenv("OPENENV_RUN_ID", raising=False)
    labels = sandbox.sandbox_labels(Path("/opt/tb2-tasks/regex-chess"))
    assert labels["openenv-tbench2-task"] == "regex-chess"
    assert labels["openenv-launcher"]  # some non-empty identity, never absent
    assert "openenv-run-id" not in labels  # omitted when unset, not ""


def test_sandbox_labels_explicit_launcher_and_run_id(monkeypatch):
    monkeypatch.setenv("OPENENV_LAUNCHER", "tao-lin")
    monkeypatch.setenv("OPENENV_RUN_ID", "tb2-grpo-0717")
    labels = sandbox.sandbox_labels(Path("/opt/tb2-tasks/regex-chess"))
    assert labels["openenv-launcher"] == "tao-lin"
    assert labels["openenv-run-id"] == "tb2-grpo-0717"


def test_create_arms_ttl_by_default():
    # The dead-man's-switch contract: creates must arm auto-stop/auto-delete,
    # or a hard-killed caller's orphans run (and bill) forever.
    sig = inspect.signature(sandbox.create_task_sandbox)
    assert sig.parameters["auto_stop_minutes"].default > 0
    assert sig.parameters["auto_delete_minutes"].default > 0


def test_keepalive_beats_then_exits_on_persistent_failure(monkeypatch):
    monkeypatch.setattr(sandbox, "_KEEPALIVE_INTERVAL_S", 0.02)

    class Stub:
        def __init__(self):
            self.beats = 0
            self.dead = False

        def refresh_activity(self):
            if self.dead:
                raise RuntimeError("sandbox deleted")
            self.beats += 1

    stub = Stub()
    sandbox._start_keepalive(stub, "regex-chess")
    deadline = time.time() + 2.0
    while stub.beats < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert stub.beats >= 3  # beats while the sandbox is alive

    stub.dead = True  # episode over, sandbox deleted -> thread must exit
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any("keepalive" in t.name for t in threading.enumerate()):
            break
        time.sleep(0.01)
    assert not any("keepalive" in t.name for t in threading.enumerate())
