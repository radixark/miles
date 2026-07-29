from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from miles.utils.workers.process_utils import kill_process_tree, launch_bound_subprocess, terminate_process_tree

_SLEEP_FOREVER = "import time; time.sleep(300)"


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(predicate, *, deadline_seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _read_when_present(path: Path, *, deadline_seconds: float = 15.0) -> str:
    assert _wait_until(lambda: path.exists() and path.read_text() != "", deadline_seconds=deadline_seconds)
    return path.read_text()


class TestTerminateProcessTree:
    def test_kills_the_child(self):
        """A launched sleeper is gone after terminate_process_tree."""
        process = launch_bound_subprocess([sys.executable, "-c", _SLEEP_FOREVER], envs={})
        assert _is_alive(process.pid)
        terminate_process_tree(process)
        assert not _is_alive(process.pid)

    def test_kills_the_grandchild_too(self, tmp_path):
        """Killing the group takes down processes the child itself spawned."""
        pid_file = tmp_path / "grandchild.pid"
        child_code = (
            "import subprocess, sys, time\n"
            f"grand = subprocess.Popen([sys.executable, '-c', {_SLEEP_FOREVER!r}])\n"
            f"open({str(pid_file)!r}, 'w').write(str(grand.pid))\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", child_code], envs={})
        grandchild_pid = int(_read_when_present(pid_file))

        terminate_process_tree(process)
        assert _wait_until(lambda: not _is_alive(grandchild_pid))

    def test_escalates_to_sigkill_when_sigterm_is_ignored(self, tmp_path):
        """A child that ignores SIGTERM is still gone after the grace period."""
        ready_file = tmp_path / "ready"
        trap_term = (
            "import pathlib, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path({str(ready_file)!r}).write_text('ready'); "
            "time.sleep(300)"
        )
        process = launch_bound_subprocess([sys.executable, "-c", trap_term], envs={})
        _read_when_present(ready_file)

        terminate_process_tree(process, sigkill_timeout=0.5)

        assert not _is_alive(process.pid)
        assert process.returncode == -signal.SIGKILL

    def test_noop_on_already_exited_process(self):
        """Terminating an exited process is safe and idempotent."""
        process = launch_bound_subprocess([sys.executable, "-c", "pass"], envs={})
        process.wait(timeout=15)
        terminate_process_tree(process)
        terminate_process_tree(process)


class TestKillProcessTree:
    def test_kills_the_child_without_a_term_grace_period(self):
        """The kill is a crash simulation, so the child gets SIGKILL, not SIGTERM."""
        trap_term = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"
        process = launch_bound_subprocess([sys.executable, "-c", trap_term], envs={})
        assert _is_alive(process.pid)

        kill_process_tree(process)

        process.wait(timeout=15)
        assert process.returncode == -signal.SIGKILL


class TestLaunchBoundSubprocess:
    def test_envs_are_merged_over_the_parent_environment(self, tmp_path):
        """Passed envs reach the child on top of the inherited environment."""
        out_file = tmp_path / "env.txt"
        code = f"import os; open({str(out_file)!r}, 'w').write(os.environ['MILES_BOUND_TEST_VAR'])"
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={"MILES_BOUND_TEST_VAR": "yes"})
        process.wait(timeout=15)
        assert out_file.read_text() == "yes"

    def _launch_intermediate_parent(self, tmp_path, *, stay_alive: bool) -> tuple[subprocess.Popen, int]:
        pid_file = tmp_path / "bound_child.pid"
        parent_code = (
            "import sys, time\n"
            "from miles.utils.workers.process_utils import launch_bound_subprocess\n"
            f"process = launch_bound_subprocess([sys.executable, '-c', {_SLEEP_FOREVER!r}], envs={{}})\n"
            f"open({str(pid_file)!r}, 'w').write(str(process.pid))\n" + ("time.sleep(300)\n" if stay_alive else "")
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_code])
        bound_child_pid = int(_read_when_present(pid_file))
        return parent, bound_child_pid

    def test_child_dies_when_parent_exits_normally(self, tmp_path):
        """The atexit hook reaps the bound child on normal parent exit."""
        parent, bound_child_pid = self._launch_intermediate_parent(tmp_path, stay_alive=False)
        parent.wait(timeout=15)
        assert _wait_until(lambda: not _is_alive(bound_child_pid))

    @pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is linux-only")
    def test_child_dies_when_parent_is_sigkilled(self, tmp_path):
        """The parent-death signal reaps the bound child even on SIGKILL."""
        parent, bound_child_pid = self._launch_intermediate_parent(tmp_path, stay_alive=True)
        assert _is_alive(bound_child_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=15)
        assert _wait_until(lambda: not _is_alive(bound_child_pid))
