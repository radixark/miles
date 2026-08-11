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

    def test_reports_what_the_cluster_said_when_it_refuses(self, monkeypatch):
        """An expired token and a missing cluster look identical unless the message is passed through."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "mine")
        monkeypatch.setattr(
            cluster_backends.subprocess,
            "run",
            lambda *args, **kwargs: _completed(returncode=1, stderr="Unauthorized"),
        )

        assert "Unauthorized" in cluster_backends.kubernetes_availability().reason

    def test_says_an_admin_has_to_install_leaderworkerset(self, monkeypatch):
        """Nothing a developer can do about it, so the message has to say who can."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "mine")
        calls = []

        def fake_run(command, *args, **kwargs):
            calls.append(command)
            return _completed(returncode=0 if "--raw" in command else 1)

        monkeypatch.setattr(cluster_backends.subprocess, "run", fake_run)

        assert "admin" in cluster_backends.kubernetes_availability().reason


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


def _completed(returncode=0, stdout="", stderr=""):
    import subprocess

    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)
