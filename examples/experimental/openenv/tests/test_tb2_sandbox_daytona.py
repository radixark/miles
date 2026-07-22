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

import pytest

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


def test_resolve_api_key_env_value_wins(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "api_key"
    key_file.write_text("dtn_from_file\n")
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn_from_env")
    monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "dtn_from_env"


def test_resolve_api_key_falls_back_to_file(monkeypatch, tmp_path: Path):
    # The launcher forwards only this path (never the value, which would be
    # echoed into driver logs via ray runtime_env); workers must read it here.
    key_file = tmp_path / "api_key"
    key_file.write_text("dtn_from_file\n")
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "dtn_from_file"  # whitespace stripped


def test_resolve_api_key_default_path_under_home(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".config" / "daytona"
    cfg.mkdir(parents=True)
    (cfg / "api_key").write_text("dtn_default\n")
    assert sandbox.resolve_api_key() == "dtn_default"


def test_resolve_api_key_errors_when_absent(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing or empty"):
        sandbox.resolve_api_key()


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
