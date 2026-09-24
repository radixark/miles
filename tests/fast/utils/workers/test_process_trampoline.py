from __future__ import annotations

import ast
import ctypes
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import miles.utils.workers.process_trampoline as process_trampoline

_TRAMPOLINE_MODULE = "miles.utils.workers.process_trampoline"


@dataclass
class _TrampolineRun:
    events: list[str] = field(default_factory=list)
    prctl_calls: list[tuple[int, int]] = field(default_factory=list)
    exec_calls: list[tuple[str, list[str]]] = field(default_factory=list)
    popen_calls: list[list[str]] = field(default_factory=list)
    exit_status: int | None = None


def _run_main(
    *,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    platform: str = "linux",
    current_parent_pid: int,
    exec_error: OSError | None = None,
) -> _TrampolineRun:
    run = _TrampolineRun()

    class _FakeLibc:
        def prctl(self, option: int, signal_number: int) -> int:
            run.events.append("prctl")
            run.prctl_calls.append((option, signal_number))
            return 0

    def fake_getppid() -> int:
        run.events.append("getppid")
        return current_parent_pid

    def fake_execvp(file: str, args: list[str]) -> None:
        run.events.append("execvp")
        run.exec_calls.append((file, args))
        if exec_error is not None:
            raise exec_error

    def fake_exit(status: int) -> None:
        run.events.append("exit")
        raise SystemExit(status)

    class _FakeProcess:
        def wait(self) -> int:
            run.events.append("wait")
            return 7

    def fake_popen(args: list[str]) -> _FakeProcess:
        run.events.append("popen")
        run.popen_calls.append(args)
        if exec_error is not None:
            raise exec_error
        return _FakeProcess()

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(ctypes, "CDLL", lambda name, use_errno=False: _FakeLibc())
    monkeypatch.setattr(os, "getppid", fake_getppid)
    monkeypatch.setattr(os, "execvp", fake_execvp)
    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(signal, "signal", lambda *_args: run.events.append("signal"))

    try:
        process_trampoline.main()
    except SystemExit as error:
        run.exit_status = error.code if isinstance(error.code, int) else 1

    return run


def _run_trampoline_process(*, expected_parent_pid: int, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", _TRAMPOLINE_MODULE, str(expected_parent_pid), *argv],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class TestTrampolineMain:
    def test_the_death_signal_is_distinct_from_graceful_sigterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parent death must reap the group without shortening normal shutdown."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            current_parent_pid=4242,
        )

        assert run.prctl_calls == [(1, signal.SIGUSR2)]

    def test_the_real_command_is_supervised_with_its_own_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Everything after the expected parent pid becomes the child's argv."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/sh", "-c", "sleep 1", ""],
            current_parent_pid=4242,
        )

        assert run.popen_calls == [["/bin/sh", "-c", "sleep 1", ""]]
        assert run.exit_status == 7

    def test_the_death_signal_is_armed_before_the_parent_is_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checking the parent first would leave a window where the death signal is never armed."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            current_parent_pid=4242,
        )

        assert run.events == ["signal", "signal", "prctl", "getppid", "popen", "wait", "exit"]

    def test_a_changed_parent_makes_it_exit_instead_of_exec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A child reparented before the signal was armed would never be reaped, so it must not run the command."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            current_parent_pid=1,
        )

        assert run.exit_status == 1
        assert run.popen_calls == []

    def test_a_failed_launch_is_reported_and_exits_with_the_shell_convention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing or unrunnable command must not look like the real command dying with status 1."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "1234", "no-such-binary", "--flag"],
            current_parent_pid=1234,
            exec_error=FileNotFoundError(2, "No such file or directory"),
        )

        assert run.events[-2:] == ["popen", "exit"]
        assert run.exit_status == 127

    def test_off_linux_it_execs_without_arming_or_checking_anything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PR_SET_PDEATHSIG is linux-only, so elsewhere the trampoline degrades to a plain exec."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            platform="darwin",
            current_parent_pid=1,
        )

        assert run.prctl_calls == []
        assert run.events == ["execvp"]
        assert run.exec_calls == [("/bin/echo", ["/bin/echo", "hi"])]


class TestTrampolineProcess:
    def test_the_exit_status_is_the_real_commands_own(self) -> None:
        """The supervisor preserves the command's exit status."""
        result = _run_trampoline_process(
            expected_parent_pid=os.getpid(),
            argv=[sys.executable, "-c", "raise SystemExit(7)"],
        )

        assert result.returncode == 7

    def test_the_command_arguments_arrive_unchanged(self) -> None:
        """Joining the argv into a shell string would expand, split or drop these arguments."""
        arguments = ["--flag", "a b", "", "$HOME", 'quote"d']
        code = "import json, sys; print(json.dumps(sys.argv[1:]))"
        result = _run_trampoline_process(
            expected_parent_pid=os.getpid(),
            argv=[sys.executable, "-c", code, *arguments],
        )

        assert result.returncode == 0
        assert json.loads(result.stdout) == arguments

    @pytest.mark.skipif(sys.platform != "linux", reason="requires Linux parent-death signals")
    def test_parent_death_releases_a_grandchild_listener(self, tmp_path: Path) -> None:
        """Abrupt actor death must not leave a server listening behind its shell."""
        ready_path = tmp_path / "port"
        server_code = """
import pathlib
import socket
import sys
import time

listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
pathlib.Path(sys.argv[1]).write_text(str(listener.getsockname()[1]))
time.sleep(60)
"""
        shell_command = shlex.join([sys.executable, "-c", server_code, str(ready_path)]) + " & wait"
        parent_code = """
import os
import subprocess
import sys

argv = [sys.executable, *sys.argv[1:]]
argv[3] = str(os.getpid())
child = subprocess.Popen(argv, start_new_session=True)
print(child.pid, flush=True)
child.wait()
"""
        parent = subprocess.Popen(
            [
                sys.executable,
                "-c",
                parent_code,
                "-m",
                _TRAMPOLINE_MODULE,
                "PARENT_PID",
                "/bin/sh",
                "-c",
                shell_command,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert parent.stdout is not None
        supervisor_pid = int(parent.stdout.readline())
        try:
            for _ in range(100):
                if ready_path.exists():
                    break
                time.sleep(0.05)
            assert ready_path.exists(), "grandchild did not start"
            port = int(ready_path.read_text())
            parent.kill()
            parent.wait(timeout=5)
            for _ in range(100):
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", port))
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                pytest.fail(f"grandchild retained port {port} after its owner died")
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            try:
                os.killpg(supervisor_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class TestTrampolineModule:
    def test_it_imports_nothing_but_the_standard_library_modules_it_needs(self) -> None:
        """The trampoline runs before every launched command, so importing miles code would be a new failure mode."""
        source = Path(process_trampoline.__file__).read_text()

        imported_modules: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module.split(".")[0])

        assert imported_modules == {"ctypes", "os", "signal", "subprocess", "sys"}
