from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.orchestrator.observer_fakes import _ObserverProbe

from miles.utils.external_utils.command_utils.helm_backend.orchestrator import observer
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)


class TestUnreadableGenerationBudget:
    def test_thirty_unreadable_generations_fail_without_trusting_a_terminal_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale success file cannot hide permanent loss of the active generation."""
        path = tmp_path / "state.json"
        OrchestratorState(status=OrchestratorStatus.EXITED, exit_code=0).write(path)
        probe = _ObserverProbe(generations=[OSError("unreachable")] * 30, pods=[])
        monkeypatch.setattr(observer, "time", SimpleNamespace(sleep=probe.sleep))

        with pytest.raises(RuntimeError, match="30 polls in a row"):
            observer.wait_for_run(
                state_file=path, read_pod=probe.read_pod, read_active_state_file=probe.read_generation
            )

        assert probe.generation_reads == 30
        assert probe.pod_reads == 0
        assert len(probe.sleeps) == 29

    def test_a_readable_generation_resets_the_consecutive_failure_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two separated outages below the limit must not accumulate into a false failure."""
        path = tmp_path / "state.json"
        probe = _ObserverProbe(
            generations=[OSError("first")] * 29 + [path] + [OSError("second")] * 30,
            pods=[observer.ObservedPod(phase="Running")],
        )
        monkeypatch.setattr(observer, "time", SimpleNamespace(sleep=probe.sleep))

        with pytest.raises(RuntimeError, match="30 polls in a row"):
            observer.wait_for_run(
                state_file=path, read_pod=probe.read_pod, read_active_state_file=probe.read_generation
            )

        assert probe.generation_reads == 60
        assert probe.pod_reads == 1
