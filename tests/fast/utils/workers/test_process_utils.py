from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import miles.utils.workers.process_utils as process_utils
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

    def test_sigkills_grandchild_after_leader_exits(self, tmp_path: Path) -> None:
        """An ignored SIGTERM is escalated after the process-group leader exits."""
        pid_file = tmp_path / "grandchild.pid"
        grandchild_code = (
            "import os, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(300)\n"
        )
        leader_code = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", leader_code], envs={})
        grandchild_pid = int(_read_when_present(pid_file))

        try:
            _terminate_process_tree(process, sigkill_timeout=0.1)
            assert _wait_until(lambda: not _is_alive(grandchild_pid), deadline_seconds=1.0)
        finally:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_sigkills_grandchild_when_leader_already_exited(self, tmp_path: Path) -> None:
        """An exited process-group leader does not prevent cleanup of a surviving descendant."""
        pid_file = tmp_path / "grandchild.pid"
        grandchild_code = (
            "import os, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(300)\n"
        )
        leader_code = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            f"pid_file = Path({str(pid_file)!r})\n"
            "while not pid_file.exists():\n"
            "    time.sleep(0.01)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", leader_code], envs={})
        grandchild_pid = int(_read_when_present(pid_file))
        process.wait(timeout=15)

        try:
            _terminate_process_tree(process, sigkill_timeout=0.1)
            assert _wait_until(lambda: not _is_alive(grandchild_pid), deadline_seconds=1.0)
        finally:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_escalates_to_sigkill_when_sigterm_is_ignored(self, tmp_path: Path) -> None:
        """A process that ignores SIGTERM is terminated with SIGKILL after the timeout."""
        ready_file = tmp_path / "ready"
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(ready_file)!r}, 'w').write('ready')\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={})
        _read_when_present(ready_file)

        _terminate_process_tree(process, sigkill_timeout=0.1)

        assert process.returncode == -signal.SIGKILL

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

    def test_child_dies_when_parent_exits_normally(self, tmp_path: Path) -> None:
        """The atexit hook reaps a bound child's descendant on normal parent exit."""
        grandchild_pid_file = tmp_path / "bound_grandchild.pid"
        grandchild_code = (
            "import os, time\n"
            f"open({str(grandchild_pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(300)\n"
        )
        child_code = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            "time.sleep(300)\n"
        )
        parent_code = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "from miles.utils.workers.process_utils import launch_bound_subprocess\n"
            f"launch_bound_subprocess([sys.executable, '-c', {child_code!r}], envs={{}})\n"
            f"pid_file = Path({str(grandchild_pid_file)!r})\n"
            "while not pid_file.exists():\n"
            "    time.sleep(0.01)\n"
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_code])
        grandchild_pid = int(_read_when_present(grandchild_pid_file))

        try:
            parent.wait(timeout=15)
            assert _wait_until(lambda: not _is_alive(grandchild_pid))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=15)
            try:
                process_group_id = os.getpgid(grandchild_pid)
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_no_python_runs_between_the_fork_and_the_exec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """preexec_fn runs python in a forked child of a threaded actor, which can deadlock before exec."""
        popen_kwargs: list[dict[str, object]] = []

        def fake_popen(argv: list[str], **kwargs: object) -> object:
            popen_kwargs.append(kwargs)
            return object()

        monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(process_utils.atexit, "register", lambda function, *args: function)

        process_utils.launch_bound_subprocess([sys.executable, "-c", "pass"], envs={})

        assert "preexec_fn" not in popen_kwargs[0]
        assert popen_kwargs[0]["start_new_session"] is True

    def test_the_launched_command_is_wrapped_in_the_trampoline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The parent pid is captured before spawning and handed to the trampoline as an argument."""
        launched_argvs: list[list[str]] = []

        def fake_popen(argv: list[str], **kwargs: object) -> object:
            launched_argvs.append(argv)
            return object()

        monkeypatch.setattr(process_utils.os, "getpid", lambda: 12345)
        monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(process_utils.atexit, "register", lambda function, *args: function)

        process_utils.launch_bound_subprocess(["/bin/sh", "-c", "true"], envs={})

        assert launched_argvs == [
            [sys.executable, "-m", "miles.utils.workers.process_trampoline", "12345", "/bin/sh", "-c", "true"]
        ]

    def test_the_trampoline_execs_the_real_command_in_place(self, tmp_path: Path) -> None:
        """An exec replaces the trampoline, so the reported pid must be the command's own process."""
        out_file = tmp_path / "pid.txt"
        code = f"import os, sys; open({str(out_file)!r}, 'w').write(f'{{os.getpid()}} {{sys.argv[0]}}')"
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={})
        process.wait(timeout=15)

        reported_pid, argv0 = _read_when_present(out_file).split(" ")
        assert int(reported_pid) == process.pid
        assert argv0 == "-c"

    @pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is linux-only")
    def test_the_trampoline_exits_when_the_parent_is_already_gone(self, tmp_path: Path) -> None:
        """Between the fork and the prctl the parent may die, and then nothing would ever reap the child."""
        marker = tmp_path / "ran"
        code = f"open({str(marker)!r}, 'w').write('ran')"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "miles.utils.workers.process_trampoline",
                "1",
                sys.executable,
                "-c",
                code,
            ]
        )
        process.wait(timeout=15)

        assert process.returncode == 1
        assert not marker.exists()

    @pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is linux-only")
    def test_child_dies_when_parent_is_sigkilled(self, tmp_path):
        """The parent-death signal reaps the bound child even on SIGKILL."""
        parent, bound_child_pid = self._launch_intermediate_parent(tmp_path, stay_alive=True)
        assert _is_alive(bound_child_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=15)
        assert _wait_until(lambda: not _is_alive(bound_child_pid))
