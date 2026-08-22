from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import subprocess

import pytest
from tests.e2e.k8s_apiserver import apiserver as apiserver_module

RESET = subprocess.CalledProcessError(1, ["docker", "pull"], stderr="read: connection reset by peer")


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(apiserver_module.time, "sleep", lambda _seconds: None)


def _install_pulls(monkeypatch: pytest.MonkeyPatch, *, failures: int) -> list[str]:
    attempts: list[str] = []

    def fake_exec_command(cmd: str, capture_output: bool = False):
        attempts.append(cmd)
        if len(attempts) <= failures:
            raise RESET
        return None

    monkeypatch.setattr(apiserver_module, "exec_command", fake_exec_command)
    return attempts


class TestTheImagesSurviveARegistryThatResets:
    def test_a_reset_connection_is_retried_rather_than_failing_the_whole_module(self, monkeypatch: pytest.MonkeyPatch):
        """The pull runs in a fixture, so one reset takes down every test the environment serves."""
        attempts = _install_pulls(monkeypatch, failures=2)

        apiserver_module._pull_image("registry.k8s.io/etcd:3.5.15-0")

        assert len(attempts) == 3

    def test_a_registry_that_stays_down_still_fails(self, monkeypatch: pytest.MonkeyPatch):
        """Retrying forever would turn a broken registry into a suite that never finishes."""
        attempts = _install_pulls(monkeypatch, failures=apiserver_module._IMAGE_PULL_ATTEMPTS)

        with pytest.raises(subprocess.CalledProcessError):
            apiserver_module._pull_image("registry.k8s.io/etcd:3.5.15-0")

        assert len(attempts) == apiserver_module._IMAGE_PULL_ATTEMPTS

    def test_a_registry_that_answers_is_asked_once(self, monkeypatch: pytest.MonkeyPatch):
        """The retry must not cost a second round trip on the ordinary path."""
        attempts = _install_pulls(monkeypatch, failures=0)

        apiserver_module._pull_image("registry.k8s.io/etcd:3.5.15-0")

        assert len(attempts) == 1

    def test_both_images_are_pulled_before_anything_is_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        """An image left to a docker run's implicit pull reports a reset registry as a container that
        would not start, and gets no retry either."""
        commands = _install_pulls(monkeypatch, failures=0)
        monkeypatch.setattr(apiserver_module, "_free_host_port", lambda: 16443)

        apiserver_module.start_apiserver(run_id="miles-k8s-test", work_dir=tmp_path)

        pulls = [cmd for cmd in commands if cmd.startswith("docker pull")]
        first_run = next(index for index, cmd in enumerate(commands) if cmd.startswith("docker run"))
        assert [apiserver_module._ETCD_IMAGE in cmd for cmd in pulls] == [True, False]
        assert [apiserver_module._APISERVER_IMAGE in cmd for cmd in pulls] == [False, True]
        assert all(commands.index(cmd) < first_run for cmd in pulls)
