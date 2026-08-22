import logging
import time

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_container,
    make_pod,
    wait_for,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import log_follower


def _pod(name="trainer-0", uid="u", container_id="docker://a", previous_container_id="", running=True):
    return make_pod(
        name=name,
        uid=uid,
        containers=(
            make_container(
                name="app",
                container_id=container_id,
                previous_container_id=previous_container_id,
                running=running,
            ),
        ),
    )


class FakeProcess:
    def __init__(self, command: list[str], lines: list[str], blocking: bool) -> None:
        self.command = command
        self.killed = False
        self.returncode = 0
        self.stdout = self._lines(lines, blocking)

    def poll(self) -> int | None:
        return 0 if self.killed else None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        self.killed = True

    def kill(self) -> None:
        self.killed = True

    def _lines(self, lines: list[str], blocking: bool):
        yield from lines
        while blocking and not self.killed:
            time.sleep(0.01)


class FakeKubectl:
    def __init__(self, monkeypatch, lines: list[str], blocking: bool) -> None:
        self.commands: list[list[str]] = []
        self.processes: list[FakeProcess] = []
        self._lines = lines
        self._blocking = blocking
        monkeypatch.setattr(log_follower.subprocess, "Popen", self._popen)

    def _popen(self, command, **kwargs) -> FakeProcess:
        self.commands.append(command)
        process = FakeProcess(command, list(self._lines), self._blocking)
        self.processes.append(process)
        return process


def _followed(monkeypatch, pods, lines=(), blocking: bool = True) -> FakeKubectl:
    fake = FakeKubectl(monkeypatch, list(lines), blocking)
    monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)
    monkeypatch.setattr(log_follower.polling, "POLL_INTERVAL_SECONDS", 0.01)
    return fake


class TestWithLogFollowing:
    def test_follows_the_container_a_pod_is_running(self, monkeypatch, caplog):
        """A run's own output is what a user launched the launcher to read."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "hello" in caplog.text)

        assert "[trainer-0/app] hello" in caplog.text
        assert fake.commands[0][:3] == ["kubectl", "logs", "trainer-0"]
        assert "--follow" in fake.commands[0]

    def test_reads_the_life_a_container_already_crashed_out_of(self, monkeypatch, caplog):
        """The log of the container that died is the only record of why the pod will not start."""
        pods = [_pod(container_id="docker://b", previous_container_id="docker://a")]
        fake = _followed(monkeypatch, pods, lines=["2026-08-10T00:00:00.1Z boom\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "(previous)" in caplog.text)

        assert any("--previous" in command for command in fake.commands)

    def test_does_not_start_a_second_stream_for_a_container_it_already_reads(self, monkeypatch, caplog):
        """Every poll sees the same container, and a stream per poll would print the log many times over."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "hello" in caplog.text)
                time.sleep(0.1)

        assert len(fake.commands) == 1

    def test_asks_only_for_what_it_has_not_read_when_a_stream_drops(self, monkeypatch, caplog):
        """A reconnect that started over would reprint the whole log every time the api server blinked."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"], blocking=False)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: len(fake.commands) >= 2)

        assert "--since-time" in fake.commands[-1]
        assert "2026-08-10T00:00:00.1Z" in fake.commands[-1]

    def test_stops_every_stream_when_the_caller_leaves(self, monkeypatch, caplog):
        """kubectl logs --follow never returns on its own, so a leaked one outlives the whole launcher."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: bool(fake.processes))

        assert all(process.killed for process in fake.processes)

    def test_does_not_reconnect_to_a_container_that_ended(self, monkeypatch, caplog):
        """kubectl returns at once on a dead container, so retrying it would spawn a process every poll."""
        fake = _followed(monkeypatch, [_pod(running=False)], lines=["2026-08-10T00:00:00.1Z bye\n"], blocking=False)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "bye" in caplog.text)
                time.sleep(0.1)

        assert len(fake.commands) == 1

    def test_does_not_read_a_crashed_life_it_already_followed_live(self, monkeypatch, caplog):
        """Replaying an hour of log under a (previous) prefix buries the crash the user came to read."""
        running = [_pod(container_id="docker://a")]
        crashed = [_pod(container_id="docker://b", previous_container_id="docker://a")]
        pods = running
        fake = _followed(monkeypatch, pods, lines=["2026-08-10T00:00:00.1Z hello\n"])
        monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: bool(fake.commands))
                pods = crashed
                monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)
                wait_for(lambda: len(fake.commands) >= 2)
                time.sleep(0.1)

        assert not any("--previous" in command for command in fake.commands)


class TestLogsCommand:
    def test_asks_for_the_timestamps_a_reconnect_needs(self):
        """--since-time is the only way back to where a dropped stream stopped, and it reads these stamps."""
        command = Kubectl.logs_command(
            namespace="rl", target="trainer-0", container="app", follow=True, previous=False, since_time=None
        )

        assert "--timestamps" in command
        assert "--since-time" not in command

    def test_does_not_follow_a_container_that_already_ended(self):
        """--follow on a dead container returns at once; asking plainly is what actually fetches its log."""
        command = Kubectl.logs_command(
            namespace="rl", target="trainer-0", container="app", follow=False, previous=True, since_time="t"
        )

        assert "--follow" not in command
        assert command[-2:] == ["--since-time", "t"]
