import json
import subprocess
from pathlib import Path

import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import NAMESPACE, RELEASE
from tests.fast.utils.soak.deploy.deploy_fakes import (
    _STATE_FILE,
    _deployment_target,
    _FakeJobKubectl,
    _FakeWorkloadKubectl,
    _workload_item,
)
from tests.utils.soak.deploy.guard import launch_guard, target_check
from tests.utils.soak.deploy.guard.launch_guard import GUARDED_UPGRADE_TIMEOUT_SECONDS, HotRestartLaunchGuard

from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.workers.cell_operations.base import StaleFaultTargetError

_JOB = RunNames.uninstall_job(release=RELEASE)


def _guard(**overrides: object) -> HotRestartLaunchGuard:
    return HotRestartLaunchGuard(target=_deployment_target(**overrides))


def _install_jobs(monkeypatch: pytest.MonkeyPatch, kubectl: _FakeJobKubectl) -> _FakeJobKubectl:
    monkeypatch.setattr(launch_guard, "run_process", kubectl)
    return kubectl


class TestHotRestartLaunchGuardCreate:
    def test_a_target_without_an_observed_state_file_is_stale(self) -> None:
        """Without the orchestrator generation there is nothing to compare the launcher's view against."""
        with pytest.raises(StaleFaultTargetError, match="state file"):
            _guard(state_file=None)


class TestHotRestartLaunchGuardBeforeDefuse:
    def test_the_observed_generation_with_unchanged_workloads_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The launcher may defuse the old orchestrator only while the workloads are still the observed ones."""
        kubectl = _FakeWorkloadKubectl(stateful_sets=[_workload_item(RELEASE + "-a", stamp=None)])
        monkeypatch.setattr(target_check, "run_process", kubectl)
        guard = _guard(stamps={RELEASE + "-a": None})

        guard.before_defuse(RELEASE, namespace=NAMESPACE, superseded_state_file=_STATE_FILE, state_file=Path("/new"))

        assert len(kubectl.calls) == 2

    @pytest.mark.parametrize(
        ("release", "namespace", "superseded"),
        [
            pytest.param("miles-other-all", NAMESPACE, _STATE_FILE, id="release"),
            pytest.param(RELEASE, "other", _STATE_FILE, id="namespace"),
            pytest.param(RELEASE, NAMESPACE, Path("/state/generation-b.json"), id="newer_generation"),
            pytest.param(RELEASE, NAMESPACE, None, id="no_generation"),
        ],
    )
    def test_a_launcher_seeing_another_generation_is_refused_before_reading_workloads(
        self, monkeypatch: pytest.MonkeyPatch, release: str, namespace: str, superseded: Path | None
    ) -> None:
        """A newer orchestrator state means another take-over already superseded the observed one."""
        kubectl = _FakeWorkloadKubectl(stateful_sets=[])
        monkeypatch.setattr(target_check, "run_process", kubectl)

        with pytest.raises(StaleFaultTargetError, match="observed orchestrator generation"):
            _guard().before_defuse(release, namespace=namespace, superseded_state_file=superseded, state_file=None)

        assert kubectl.calls == []

    def test_changed_workloads_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even with the same state file a restamped workload means the observation is stale."""
        kubectl = _FakeWorkloadKubectl(stateful_sets=[_workload_item(RELEASE + "-a", stamp="t9")])
        monkeypatch.setattr(target_check, "run_process", kubectl)

        with pytest.raises(StaleFaultTargetError, match="changed since observation"):
            _guard(stamps={RELEASE + "-a": None}).before_defuse(
                RELEASE, namespace=NAMESPACE, superseded_state_file=_STATE_FILE, state_file=None
            )


class TestHotRestartLaunchGuardDeleteUninstallJob:
    def test_the_observed_job_is_deleted_with_a_uid_precondition_and_awaited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the exact job incarnation observed may be deleted, and the launch waits until it is gone."""
        kubectl = _install_jobs(monkeypatch, _FakeJobKubectl(current_uid="uid-uninstall"))

        _guard().delete_uninstall_job(_JOB, namespace=NAMESPACE)

        assert kubectl.verbs() == ["get", "delete", "wait"]
        get, delete, wait = kubectl.calls
        assert get["argv"][:4] == ["kubectl", "get", "job", _JOB] and "--ignore-not-found" in get["argv"]
        assert delete["argv"][3] == f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{_JOB}"
        options = json.loads(delete["input"])
        assert options["preconditions"] == {"uid": "uid-uninstall"}
        assert options["propagationPolicy"] == "Foreground"
        assert wait["argv"][:4] == ["kubectl", "wait", "--for=delete", f"job/{_JOB}"]
        assert all(call["check"] for call in kubectl.calls)

    def test_an_absent_job_observed_absent_deletes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job that was not there at observation and is not there now needs no delete."""
        kubectl = _install_jobs(monkeypatch, _FakeJobKubectl(current_uid=None))

        _guard(uninstall_job_uid=None).delete_uninstall_job(_JOB, namespace=NAMESPACE)

        assert kubectl.verbs() == ["get"]

    @pytest.mark.parametrize(
        ("observed", "current"),
        [
            pytest.param("uid-uninstall", "uid-other", id="replaced"),
            pytest.param("uid-uninstall", None, id="deleted_since"),
            pytest.param(None, "uid-new", id="created_since"),
        ],
    )
    def test_a_job_that_changed_since_observation_is_refused_without_deleting(
        self, monkeypatch: pytest.MonkeyPatch, observed: str | None, current: str | None
    ) -> None:
        """A different uninstall job belongs to a newer take-over and must be left alone."""
        kubectl = _install_jobs(monkeypatch, _FakeJobKubectl(current_uid=current))

        with pytest.raises(StaleFaultTargetError, match="uninstall job changed"):
            _guard(uninstall_job_uid=observed).delete_uninstall_job(_JOB, namespace=NAMESPACE)

        assert kubectl.verbs() == ["get"]

    @pytest.mark.parametrize(
        ("name", "namespace"), [pytest.param("miles-other-all-uninstall", NAMESPACE, id="name"), (_JOB, "other")]
    )
    def test_a_job_of_another_release_is_refused_before_any_kubectl(
        self, monkeypatch: pytest.MonkeyPatch, name: str, namespace: str
    ) -> None:
        """The launcher may only delete this release's uninstall job in this namespace."""
        kubectl = _install_jobs(monkeypatch, _FakeJobKubectl(current_uid="uid-uninstall"))

        with pytest.raises(StaleFaultTargetError, match="another release"):
            _guard().delete_uninstall_job(name, namespace=namespace)

        assert kubectl.calls == []

    def test_a_failed_wait_is_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job that never finishes deleting must stop the launch instead of racing it."""
        _install_jobs(monkeypatch, _FakeJobKubectl(current_uid="uid-uninstall", failing_verb="wait"))

        with pytest.raises(subprocess.CalledProcessError):
            _guard().delete_uninstall_job(_JOB, namespace=NAMESPACE)


class TestHotRestartLaunchGuardUpgrade:
    def test_the_upgrade_of_this_release_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guarded upgrade forwards every argument and adds a finite timeout."""
        calls: list[dict] = []
        monkeypatch.setattr(launch_guard.Helm, "upgrade", lambda **kwargs: calls.append(kwargs))

        _guard().upgrade(release=RELEASE, namespace=NAMESPACE, chart="chart", values_files=["v.yaml"], ci_run=True)

        assert calls == [
            {
                "release": RELEASE,
                "namespace": NAMESPACE,
                "chart": "chart",
                "values_files": ["v.yaml"],
                "ci_run": True,
                "timeout": GUARDED_UPGRADE_TIMEOUT_SECONDS,
            }
        ]

    @pytest.mark.parametrize(("release", "namespace"), [("miles-other-all", NAMESPACE), (RELEASE, "other")])
    def test_an_upgrade_of_another_release_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, release: str, namespace: str
    ) -> None:
        """Upgrading a release other than the observed one would take over a run nobody watched."""
        calls: list[dict] = []
        monkeypatch.setattr(launch_guard.Helm, "upgrade", lambda **kwargs: calls.append(kwargs))

        with pytest.raises(StaleFaultTargetError, match="another release"):
            _guard().upgrade(release=release, namespace=namespace, chart="chart", values_files=[], ci_run=False)

        assert calls == []
