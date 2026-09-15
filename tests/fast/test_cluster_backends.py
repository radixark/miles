import subprocess

import pytest
from tests.fast import cluster_backends


class TestKubernetesAvailability:
    def test_names_the_missing_tool_rather_than_just_refusing(self, monkeypatch):
        """A developer reading "unavailable" learns nothing; naming kubectl tells them what to install."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: None)

        assert "kubectl" in cluster_backends.kubernetes_availability().reason

    def test_points_at_the_runbook_when_no_namespace_is_set(self, monkeypatch):
        """Running in whatever namespace happened to be current is how someone else's run gets deleted."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv(cluster_backends.NAMESPACE_ENV_VAR, raising=False)

        reason = cluster_backends.kubernetes_availability().reason

        assert cluster_backends.NAMESPACE_ENV_VAR in reason
        assert "cluster-backend" in reason

    def test_names_the_variable_the_caller_reads_rather_than_its_own(self, monkeypatch):
        """Two callers feed this from two different variables, so a hardcoded hint sends one of them to the wrong knob."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")

        reason = cluster_backends.kubernetes_availability_of_namespace(
            "", namespace_source="MILES_SCRIPT_NAMESPACE"
        ).reason

        assert "MILES_SCRIPT_NAMESPACE" in reason
        assert cluster_backends.NAMESPACE_ENV_VAR not in reason

    def test_reports_what_the_cluster_said_when_it_refuses(self, monkeypatch):
        """An expired token and a missing cluster look identical unless the message is passed through."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(returncode=1, stderr="Unauthorized"))

        assert "Unauthorized" in cluster_backends.kubernetes_availability().reason

    def test_says_an_admin_has_to_install_leaderworkerset(self, monkeypatch):
        """Nothing a developer can do about it, so the message has to say who can."""
        _fake_kubectl(
            monkeypatch,
            lambda argv: _completed(returncode=1 if cluster_backends.LEADER_WORKER_SET_API_PATH in argv else 0),
        )

        assert "admin" in cluster_backends.kubernetes_availability().reason

    def test_asks_the_api_for_leaderworkerset_instead_of_reading_the_crd(self, monkeypatch):
        """The workbench holds a namespaced Role, so a cluster-scoped read reports "not installed" for a cluster that has it."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        cluster_backends.kubernetes_availability()

        assert any(cluster_backends.LEADER_WORKER_SET_API_PATH in argv for argv in calls)
        assert not any("crd" in argv or "customresourcedefinitions" in argv for argv in calls)

    def test_refuses_when_the_account_may_not_delete_pods(self, monkeypatch):
        """Injecting a fault deletes a pod, so discovering the missing verb mid-soak wastes the whole run."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="" if "delete" in argv else "yes"))

        reason = cluster_backends.kubernetes_availability().reason

        assert "delete" in reason and "pods" in reason

    def test_bounds_every_call_so_an_unreachable_cluster_cannot_hang_the_probe(self, monkeypatch):
        """kubectl against a black-holed endpoint blocks forever, and the probe runs before anything can time it out."""
        timeouts = []

        def fake_run_process(argv, *, capture_output, check, input=None, timeout=None):
            timeouts.append(timeout)
            return _completed(stdout="yes")

        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "mine")
        monkeypatch.setattr(cluster_backends, "run_process", fake_run_process)

        cluster_backends.kubernetes_availability()

        assert timeouts and all(t == cluster_backends.KUBECTL_TIMEOUT_SECONDS for t in timeouts)

    def test_accepts_a_cluster_that_serves_the_api_and_grants_every_verb(self, monkeypatch):
        """The negative cases prove nothing unless the positive one still passes."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        availability = cluster_backends.kubernetes_availability()

        assert availability.available
        assert "mine" in availability.reason


class TestRequireBackend:
    def test_skips_with_the_reason_attached(self, monkeypatch):
        """A skip nobody can act on is as unhelpful as a silent pass."""
        monkeypatch.setattr(
            cluster_backends,
            "_AVAILABILITY",
            {"kubernetes": lambda: cluster_backends.BackendAvailability(False, "no cluster today")},
        )

        with pytest.raises(BaseException, match="no cluster today"):
            cluster_backends.require_backend("kubernetes")

    def test_hands_back_the_namespace_when_the_backend_is_usable(self, monkeypatch):
        """Tests install into it, so they need it and must not read the environment themselves."""
        monkeypatch.setattr(
            cluster_backends,
            "_AVAILABILITY",
            {"kubernetes": lambda: cluster_backends.BackendAvailability(True, "fine")},
        )
        monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "miles-e2e-mine")

        assert cluster_backends.require_backend("kubernetes") == "miles-e2e-mine"


class TestBothBackends:
    @cluster_backends.both_backends
    def test_runs_once_per_backend(self, cluster_backend):
        """One test body covering both is the point; a parallel copy would only test the copy."""
        assert cluster_backend in ("ray", "kubernetes")


def _fake_kubectl(monkeypatch, respond) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run_process(argv, *, capture_output, check, input=None, timeout=None):
        calls.append(argv)
        return respond(argv)

    monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "mine")
    monkeypatch.setattr(cluster_backends, "run_process", fake_run_process)
    return calls


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)
