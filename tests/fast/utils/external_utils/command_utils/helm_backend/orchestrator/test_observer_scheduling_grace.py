from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.orchestrator.observer_fakes import _ObserverProbe

from miles.utils.external_utils.command_utils.helm_backend.orchestrator import observer


class TestMissingUnstartedPodBudget:
    @pytest.mark.parametrize("polls", [0, 3, 29, 30, 31])
    def test_an_unstarted_pod_has_a_finite_thirty_poll_grace(self, polls: int) -> None:
        """Scheduling gets a longer grace than a started run, but never an infinite one."""
        outcome = observer._compute_run_outcome(
            state=None, observed=None, missing_polls=polls, dead_polls=0, failing_polls=0
        )

        if polls < 30:
            assert outcome is None
        else:
            assert outcome is not None
            assert outcome.exit_code == 1
            assert f"gone for {polls} polls" in outcome.reason

    def test_a_returning_pod_resets_the_missing_poll_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pod that appears during scheduling starts a new disappearance interval."""
        path = tmp_path / "state.json"
        probe = _ObserverProbe(
            generations=[path] * 61,
            pods=[None] * 29 + [observer.ObservedPod(phase="Pending")] + [None] * 30,
        )
        monkeypatch.setattr(observer, "time", SimpleNamespace(sleep=probe.sleep))

        outcome = observer.wait_for_run(
            state_file=path, read_pod=probe.read_pod, read_active_state_file=probe.read_generation
        )

        assert outcome.exit_code == 1
        assert probe.pod_reads == 60
        assert probe.generation_reads == 61

    def test_a_new_generation_gets_its_own_scheduling_grace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A takeover must not inherit missing-pod evidence from the replaced generation."""
        old, new = tmp_path / "old.json", tmp_path / "new.json"
        probe = _ObserverProbe(generations=[old] * 29 + [new] * 31, pods=[None] * 59)
        monkeypatch.setattr(observer, "time", SimpleNamespace(sleep=probe.sleep))

        outcome = observer.wait_for_run(
            state_file=old, read_pod=probe.read_pod, read_active_state_file=probe.read_generation
        )

        assert outcome.exit_code == 1
        assert probe.pod_reads == 59
