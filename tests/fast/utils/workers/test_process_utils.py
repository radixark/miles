from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from miles.utils.workers.process_utils import _terminate_process_tree, launch_bound_subprocess

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
        """A launched sleeper is gone after _terminate_process_tree."""
        process = launch_bound_subprocess([sys.executable, "-c", _SLEEP_FOREVER], envs={})
        assert _is_alive(process.pid)
        _terminate_process_tree(process)
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

        _terminate_process_tree(process)
        assert _wait_until(lambda: not _is_alive(grandchild_pid))

    def test_noop_on_already_exited_process(self):
        """Terminating an exited process is safe and idempotent."""
        process = launch_bound_subprocess([sys.executable, "-c", "pass"], envs={})
        process.wait(timeout=15)
        _terminate_process_tree(process)
        _terminate_process_tree(process)


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
