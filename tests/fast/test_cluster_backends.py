import subprocess

import pytest
from tests.fast import cluster_backends

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend


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


class TestCreateBackendForRun:
    def test_a_backend_the_lane_asked_for_and_cannot_reach_is_a_failure(self, monkeypatch):
        """The ft e2e entries are bare python3 with no pytest to skip into, and exiting 0 would report green."""
        monkeypatch.setattr(
            cluster_backends,
            "availability_of_run",
            lambda *, config: cluster_backends.BackendAvailability(False, "no cluster today"),
        )

        with pytest.raises(AssertionError, match="no cluster today"):
            cluster_backends.create_backend_for_run(_run_config(ClusterBackend.KUBERNETES))

    def test_a_reachable_backend_lets_the_run_proceed(self, monkeypatch):
        """The gate must be invisible on the path the run is meant to take."""
        monkeypatch.setattr(
            cluster_backends,
            "availability_of_run",
            lambda *, config: cluster_backends.BackendAvailability(True, "fine"),
        )

        cluster_backends.create_backend_for_run(_run_config(ClusterBackend.RAY))


class TestAvailabilityOfRun:
    def test_the_namespace_comes_from_the_run_config_not_the_fast_tests_variable(self, monkeypatch):
        """Regression: the workbench sets MILES_SCRIPT_NAMESPACE only, so reading the other one skipped everything."""
        monkeypatch.delenv(cluster_backends.NAMESPACE_ENV_VAR, raising=False)
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        availability = cluster_backends.availability_of_run(config=_run_config(ClusterBackend.KUBERNETES))

        assert availability.available, availability.reason
        assert all("miles-e2e" in argv for argv in calls if "can-i" in argv)

    def test_a_run_without_a_namespace_names_the_variable_the_run_reads(self, monkeypatch):
        """kubectl would otherwise act on whatever namespace the kubeconfig happens to point at."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        reason = cluster_backends.availability_of_run(
            config=_run_config(ClusterBackend.KUBERNETES, namespace="")
        ).reason

        assert cluster_backends.RUN_NAMESPACE_ENV_VAR in reason

    def test_a_ray_run_is_probed_by_importing_ray(self, monkeypatch):
        """A ray lane must not be gated on kubectl, helm or a cluster it never talks to."""
        monkeypatch.setattr(
            cluster_backends, "ray_availability", lambda: cluster_backends.BackendAvailability(True, "importable")
        )

        assert cluster_backends.availability_of_run(config=_run_config(ClusterBackend.RAY)).available


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


def _run_config(backend: ClusterBackend, *, namespace: str = "miles-e2e") -> command_utils.ExecuteTrainConfig:
    return command_utils.ExecuteTrainConfig(cluster_backend=backend, namespace=namespace)


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)
