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
