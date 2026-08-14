from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

import miles.utils.workers.process_utils as process_utils
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
            terminate_process_tree(process, sigkill_timeout=0.1)
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
            terminate_process_tree(process, sigkill_timeout=0.1)
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

        terminate_process_tree(process, sigkill_timeout=0.1)

        assert process.returncode == -signal.SIGKILL

    def _launch_slow_sigterm_exiter(self, tmp_path: Path) -> subprocess.Popen:
        ready_file = tmp_path / "ready"
        code = (
            "import os, signal, time\n"
            "def handle(signum, frame):\n"
            "    time.sleep(1.0)\n"
            "    os._exit(7)\n"
            "signal.signal(signal.SIGTERM, handle)\n"
            f"open({str(ready_file)!r}, 'w').write('ready')\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={})
        _read_when_present(ready_file)
        return process

    def test_a_generous_sigkill_timeout_lets_a_slow_sigterm_handler_exit_cleanly(self, tmp_path: Path) -> None:
        """A slow SIGTERM handler reaches its own exit code when the grace period is long enough."""
        process = self._launch_slow_sigterm_exiter(tmp_path)

        terminate_process_tree(process, sigkill_timeout=30.0)

        assert process.returncode == 7

    def test_a_short_sigkill_timeout_escalates_before_the_sigterm_handler_finishes(self, tmp_path: Path) -> None:
        """The same slow SIGTERM handler is cut short by SIGKILL when the grace period is tiny."""
        process = self._launch_slow_sigterm_exiter(tmp_path)

        terminate_process_tree(process, sigkill_timeout=0.05)

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

    def test_launch_captures_parent_pid_for_preexec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The PID captured before spawning is supplied to the pre-exec setup."""
        preexec_functions: list[Callable[[], None]] = []
        received_parent_pids: list[int] = []

        def fake_popen(argv: list[str], **kwargs: object) -> object:
            preexec_function = kwargs["preexec_fn"]
            assert callable(preexec_function)
            preexec_functions.append(preexec_function)
            return object()

        def fake_register(function: Callable[..., object], *args: object) -> Callable[..., object]:
            return function

        def fake_set_parent_death_signal(*, expected_parent_pid: int) -> None:
            received_parent_pids.append(expected_parent_pid)

        monkeypatch.setattr(process_utils.os, "getpid", lambda: 12345)
        monkeypatch.setattr(process_utils, "_LIBC", object())
        monkeypatch.setattr(process_utils, "_set_parent_death_signal", fake_set_parent_death_signal)
        monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(process_utils.atexit, "register", fake_register)

        process_utils.launch_bound_subprocess([sys.executable, "-c", "pass"], envs={})
        preexec_functions[0]()

        assert received_parent_pids == [12345]

    def test_preexec_exits_if_parent_changed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-exec setup exits after registration when the captured parent is gone."""
        prctl_arguments: list[tuple[int, signal.Signals]] = []

        class FakeLibc:
            def prctl(self, option: int, signal_number: signal.Signals) -> None:
                prctl_arguments.append((option, signal_number))

        def fake_getppid() -> int:
            assert prctl_arguments == [(process_utils._PR_SET_PDEATHSIG, signal.SIGKILL)]
            return 67890

        def fake_exit(status: int) -> None:
            raise SystemExit(status)

        monkeypatch.setattr(process_utils, "_LIBC", FakeLibc())
        monkeypatch.setattr(process_utils.os, "getppid", fake_getppid)
        monkeypatch.setattr(process_utils.os, "_exit", fake_exit)

        with pytest.raises(SystemExit) as error:
            process_utils._set_parent_death_signal(expected_parent_pid=12345)

        assert error.value.code == 1

    @pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is linux-only")
    def test_child_dies_when_parent_is_sigkilled(self, tmp_path):
        """The parent-death signal reaps the bound child even on SIGKILL."""
        parent, bound_child_pid = self._launch_intermediate_parent(tmp_path, stay_alive=True)
        assert _is_alive(bound_child_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=15)
        assert _wait_until(lambda: not _is_alive(bound_child_pid))
