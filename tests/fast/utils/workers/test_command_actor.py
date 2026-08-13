import os
import shlex
import threading
import time
from pathlib import Path

import pytest

from miles.utils.http_utils import MILES_HOST_IP_ENV
from miles.utils.misc import get_current_node_ip
from miles.utils.test_utils import fault_injector
from miles.utils.workers import process_utils
from miles.utils.workers.command_actor import CommandActor


class _FakeExit:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.event = threading.Event()
        self.codes: list[int] = []
        monkeypatch.setattr(os, "_exit", self._exit)

    def _exit(self, code: int) -> None:
        self.codes.append(code)
        self.event.set()

    def wait(self) -> None:
        assert self.event.wait(timeout=10)


class TestRun:
    def test_launches_subprocess_with_envs_merged_over_parent_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The command sees the given envs, inherited parent vars, and envs override same-named parent vars."""
        fake_exit = _FakeExit(monkeypatch)
        monkeypatch.setenv("COMMAND_ACTOR_PARENT_VAR", "parent")
        monkeypatch.setenv("COMMAND_ACTOR_OVERRIDDEN_VAR", "from-parent")
        output_path = tmp_path / "output.txt"

        CommandActor().run(
            cmd=(
                'printf "%s %s %s" "$COMMAND_ACTOR_TEST_VAR" "$COMMAND_ACTOR_PARENT_VAR" '
                f'"$COMMAND_ACTOR_OVERRIDDEN_VAR" > {output_path}'
            ),
            envs={"COMMAND_ACTOR_TEST_VAR": "hello", "COMMAND_ACTOR_OVERRIDDEN_VAR": "from-envs"},
        )
        fake_exit.wait()

        assert output_path.read_text() == "hello parent from-envs"

    def test_rejects_second_run(self, monkeypatch: pytest.MonkeyPatch):
        """Calling run a second time on the same actor is rejected."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="true", envs={})

        with pytest.raises(AssertionError):
            actor.run(cmd="true", envs={})
        fake_exit.wait()

    def test_a_launch_error_propagates_without_consuming_the_actor(self, monkeypatch: pytest.MonkeyPatch):
        """A failed launch surfaces to the caller and leaves the actor free to launch again."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        thread_creations: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def _fail_to_launch(argv: list[str], *, envs: dict[str, str]) -> None:
            raise OSError("cannot launch")

        class _FakeThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                thread_creations.append((args, kwargs))

            def start(self) -> None:
                pass

        with pytest.MonkeyPatch.context() as failing_launch:
            failing_launch.setattr(process_utils, "launch_bound_subprocess", _fail_to_launch)
            failing_launch.setattr(threading, "Thread", _FakeThread)
            with pytest.raises(OSError):
                actor.run(cmd="true", envs={})

        assert thread_creations == []

        actor.run(cmd="exit 7", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [7]


class TestLifecycleBinding:
    def test_exits_actor_process_with_zero_on_subprocess_success(self, monkeypatch: pytest.MonkeyPatch):
        """The actor process exits with code 0 when the subprocess succeeds."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd="true", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [0]

    def test_exits_actor_process_with_subprocess_returncode_on_failure(self, monkeypatch: pytest.MonkeyPatch):
        """The actor process exits with the subprocess returncode when it fails."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd="exit 7", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [7]

    def test_exits_actor_process_with_one_on_signal_killed_subprocess(self, monkeypatch: pytest.MonkeyPatch):
        """A signal-killed subprocess (negative returncode) maps to exit code 1."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd='kill -TERM "$$"', envs={})
        fake_exit.wait()

        assert fake_exit.codes == [1]

    def test_does_not_exit_while_subprocess_is_running(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """The actor keeps running until the subprocess actually exits."""
        fake_exit = _FakeExit(monkeypatch)
        flag_path = tmp_path / "flag"

        CommandActor().run(cmd=f'while [ ! -f "{flag_path}" ]; do sleep 0.01; done', envs={})

        assert not fake_exit.event.wait(timeout=0.3)
        flag_path.touch()
        fake_exit.wait()
        assert fake_exit.codes == [0]


class TestShutdown:
    def test_shutdown_stops_the_subprocess_without_the_crash_exit(self, monkeypatch: pytest.MonkeyPatch):
        """A deliberate shutdown must not look like a crash to the babysitter."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="sleep 300", envs={})

        actor.shutdown()

        actor._process.wait(timeout=10)
        assert not fake_exit.event.wait(timeout=0.5)

    def test_shutdown_before_run_is_a_noop(self):
        """Tearing down an actor that never launched anything is safe."""
        CommandActor().shutdown()

    def test_shutdown_before_run_does_not_disable_a_later_run(self, monkeypatch: pytest.MonkeyPatch):
        """A shutdown of a never-launched actor must not suppress the crash exit of a later run."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()

        actor.shutdown()
        actor.run(cmd="exit 7", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [7]

    def test_shutdown_twice_is_safe(self, monkeypatch: pytest.MonkeyPatch):
        """A second shutdown of an already-stopped subprocess is a harmless no-op."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="sleep 300", envs={})

        actor.shutdown()
        actor._process.wait(timeout=10)
        actor.shutdown()

        assert not fake_exit.event.wait(timeout=0.5)

    def test_shutdown_after_the_subprocess_exited_on_its_own_is_safe(self, monkeypatch: pytest.MonkeyPatch):
        """Shutting down after a natural subprocess exit must not raise."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="true", envs={})
        fake_exit.wait()

        actor.shutdown()


class TestKillSubprocess:
    def test_killing_the_subprocess_surfaces_as_the_actor_crash_exit(self, monkeypatch: pytest.MonkeyPatch):
        """A killed subprocess must take the actor down, as a real crash would."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="sleep 300", envs={})

        actor.kill_subprocess()

        fake_exit.wait()
        assert fake_exit.codes == [1]

    def test_kill_before_run_is_rejected(self):
        """A crash injection into an actor with no subprocess is a caller bug."""
        with pytest.raises(AssertionError):
            CommandActor().kill_subprocess()


class TestInjectFault:
    def test_a_sigkill_reaches_the_worker_process_group(self, monkeypatch: pytest.MonkeyPatch):
        """Crashing a worker must include every subprocess that belongs to that worker."""
        killed: list[int] = []
        monkeypatch.setattr(process_utils, "kill_process", _refuse_to_kill)
        monkeypatch.setattr(process_utils, "kill_process_tree", lambda process: killed.append(process.pid))
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        actor.inject_fault("sigkill")

        assert killed == [4321]

    def test_a_sigkill_does_not_leave_a_spawned_engine_process_behind(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A worker crash must not orphan an engine child that can retain its GPU allocation."""
        fake_exit = _FakeExit(monkeypatch)
        child_pid_path = tmp_path / "child-pid"
        actor = CommandActor()
        actor.run(cmd=f"sleep 300 & printf '%s' \"$!\" > {shlex.quote(str(child_pid_path))}; wait", envs={})
        _wait_for_path(child_pid_path)
        child_pid = int(child_pid_path.read_text())

        try:
            actor.inject_fault("sigkill")

            fake_exit.wait()
            _wait_for_process_exit(child_pid)
        finally:
            process_utils.kill_process_tree(actor._process)

    @pytest.mark.parametrize("mode", ["exit", "segfault", "deadlock"])
    def test_every_other_failure_mode_is_rejected(self, monkeypatch: pytest.MonkeyPatch, mode: str):
        """A process exits, segfaults and deadlocks from the inside; no signal an outsider sends reproduces that."""
        monkeypatch.setattr(process_utils, "kill_process_tree", _refuse_to_kill)
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        with pytest.raises(AssertionError, match="only sigkill"):
            actor.inject_fault(mode)

    def test_an_unknown_mode_is_rejected(self):
        """A misspelt mode must not be waved through as some default crash."""
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        with pytest.raises(ValueError):
            actor.inject_fault("nuke")

    def test_the_actor_process_survives_the_injection(self, monkeypatch: pytest.MonkeyPatch):
        """Production loses the engine, not its supervisor, so crashing the actor would be the wrong fault."""
        monkeypatch.setattr(fault_injector, "inject_fault", _refuse_to_inject)
        monkeypatch.setattr(process_utils, "kill_process_tree", lambda process: None)
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        actor.inject_fault("sigkill")

    def test_an_actor_without_a_subprocess_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """Falling back to killing the actor would inject a fault production never produces."""
        monkeypatch.setattr(fault_injector, "inject_fault", _refuse_to_inject)

        with pytest.raises(AssertionError, match="no subprocess"):
            CommandActor().inject_fault("sigkill")


class _FakeProcess:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid


def _refuse_to_inject(mode: str) -> None:
    raise AssertionError(f"the actor process must not crash itself, but {mode} was injected into it")


def _refuse_to_kill(process) -> None:
    raise AssertionError(f"no kill was expected, but pid {process.pid} was killed")


def _wait_for_path(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def _wait_for_process_exit(pid: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    raise AssertionError(f"process {pid} survived the worker fault")


class TestNodeAddress:
    def test_the_node_ip_the_actor_reports_honours_the_host_ip_override(self, monkeypatch: pytest.MonkeyPatch):
        """The address a worker publishes to its peers is the one its deployment pinned through MILES_HOST_IP."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.20.30.40")

        assert CommandActor()._get_node_ip() == "10.20.30.40"

    def test_the_node_ip_the_actor_reports_falls_back_to_the_ray_node_ip(self, monkeypatch: pytest.MonkeyPatch):
        """Without an override the worker publishes the address ray placed it on."""
        monkeypatch.delenv(MILES_HOST_IP_ENV, raising=False)

        assert CommandActor()._get_node_ip() == get_current_node_ip()
