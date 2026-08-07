from __future__ import annotations

from pathlib import Path

import pytest

from miles.utils.workers.worker_provider.kubernetes.helm import env


class TestNamespaceDiscovery:
    def test_prefers_the_namespace_the_environment_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A driver run outside a pod has no service account file to read."""
        monkeypatch.setenv(env.NAMESPACE_ENV_VAR, "team-b")

        assert env.current_namespace() == "team-b"

    def test_reads_the_namespace_of_its_own_pod(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """In a pod nobody passes the namespace in, but the service account always says it."""
        monkeypatch.delenv(env.NAMESPACE_ENV_VAR, raising=False)
        namespace_file = tmp_path / "namespace"
        namespace_file.write_text("team-c\n")
        monkeypatch.setattr(env, "NAMESPACE_FILE", namespace_file)

        assert env.current_namespace() == "team-c"

    def test_refuses_to_guess(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Guessing 'default' would watch someone else's pods and heal them."""
        monkeypatch.delenv(env.NAMESPACE_ENV_VAR, raising=False)
        monkeypatch.setattr(env, "NAMESPACE_FILE", tmp_path / "missing")

        with pytest.raises(AssertionError, match=env.NAMESPACE_ENV_VAR):
            env.current_namespace()


class TestReleaseDiscovery:
    def test_reads_the_release_the_chart_told_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The orchestrator cannot recompute the release name, because its run uuid is its own."""
        monkeypatch.setenv(env.RELEASE_ENV_VAR, "miles-run-260805")

        assert env.current_release() == "miles-run-260805"

    def test_refuses_to_guess_the_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A guessed release selects no pods at all, and the run would wait forever for its cells."""
        monkeypatch.delenv(env.RELEASE_ENV_VAR, raising=False)

        with pytest.raises(AssertionError, match=env.RELEASE_ENV_VAR):
            env.current_release()


class TestLabelKeyDiscovery:
    def test_defaults_to_the_labels_a_leader_worker_set_writes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bundled charts deploy LeaderWorkerSets, so no override is the common case."""
        for env_var in env.LABEL_KEY_ENV_VARS.values():
            monkeypatch.delenv(env_var, raising=False)

        assert env.current_label_keys().pool_id == "leaderworkerset.sigs.k8s.io/name"

    def test_lets_a_platform_name_its_own_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A platform that already labels its pods says which key means what instead of relabelling."""
        monkeypatch.setenv(env.LABEL_KEY_ENV_VARS["pool_id"], "platform.example/group")

        assert env.current_label_keys().pool_id == "platform.example/group"
