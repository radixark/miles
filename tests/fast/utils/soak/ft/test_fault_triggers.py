from pathlib import Path
from typing import Any

import pytest
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.ft import fault_triggers
from tests.utils.soak.ft.types import FaultTrigger

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

_TIMER = FaultTrigger.TIMER
_HOOK = FaultTrigger.HOOK


class TestResolve:
    @pytest.mark.parametrize(
        "requested,has_real_rollout,expected",
        [
            (None, True, {_TIMER, _HOOK}),
            ([], True, {_TIMER, _HOOK}),
            (None, False, {_TIMER}),
            ([_TIMER], True, {_TIMER}),
            ([_TIMER], False, {_TIMER}),
            ([_HOOK], True, {_HOOK}),
            ([_HOOK, _TIMER, _HOOK], True, {_TIMER, _HOOK}),
        ],
        ids=["default", "empty", "fake-default", "timer", "fake-timer", "hook", "both"],
    )
    def test_the_requested_or_default_triggers_are_resolved(
        self, requested: list[FaultTrigger] | None, has_real_rollout: bool, expected: set[FaultTrigger]
    ) -> None:
        """A real rollout draws both by default and a fake rollout silently narrows the default to timer."""
        assert fault_triggers.resolve(requested, has_real_rollout=has_real_rollout) == frozenset(expected)

    @pytest.mark.parametrize("requested", [[_HOOK], [_TIMER, _HOOK]], ids=["hook", "both"])
    def test_explicit_hook_faults_without_rollout_engines_are_refused(self, requested: list[FaultTrigger]) -> None:
        """No update reaches a trainer hook without engines, so asking for hooks explicitly must fail loudly."""
        with pytest.raises(AssertionError, match="hook-triggered faults could only expire"):
            fault_triggers.resolve(requested, has_real_rollout=False)


class TestTestNameSuffixAndTrainArgs:
    @pytest.mark.parametrize(
        "triggers,suffix",
        [({_TIMER, _HOOK}, ""), ({_TIMER}, "_timer"), ({_HOOK}, "_hook")],
        ids=["default", "timer", "hook"],
    )
    def test_only_a_non_default_trigger_set_changes_the_test_name(
        self, triggers: set[FaultTrigger], suffix: str
    ) -> None:
        """The default keeps the historical dump directory while a subset gets one of its own."""
        assert fault_triggers.compute_test_name_suffix(frozenset(triggers)) == suffix

    def test_hook_faults_lengthen_the_weight_update_timeout(self) -> None:
        """A hook delay plus recovery inside an update needs a timeout longer than the default."""
        assert fault_triggers.compute_hook_train_args(frozenset({_HOOK})) == "--update-weights-timeout 600 "
        assert fault_triggers.compute_hook_train_args(frozenset({_TIMER, _HOOK})) == "--update-weights-timeout 600 "
        assert fault_triggers.compute_hook_train_args(frozenset({_TIMER})) == ""


class TestAssertHookEvidence:
    @pytest.fixture
    def checked(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
        calls: list[tuple[str, Any]] = []
        training_events = [object()]

        def _read(events: list[SoakEvent], *, dump_dir: str | Path) -> list[object]:
            calls.append(("read", dump_dir))
            return training_events

        def _recorder(name: str) -> Any:
            def _check(events: list[SoakEvent], *, training_events: list[object]) -> None:
                calls.append((name, training_events))

            return _check

        monkeypatch.setattr(fault_triggers, "read_training_events", _read)
        for name in ("assert_hook_dispatches", "assert_p2p_receiver_failures", "assert_trainer_peers_progress"):
            monkeypatch.setattr(fault_triggers, name, _recorder(name))
        return calls

    @staticmethod
    def _check(triggers: set[FaultTrigger], *, ft_components: tuple[str, ...], backend: ClusterBackend) -> None:
        fault_triggers.assert_hook_evidence(
            frozenset(triggers),
            ft_components=ft_components,
            config=command_utils.ExecuteTrainConfig(cluster_backend=backend, namespace="ns"),
            events=[],
            dump_dir="/dump",
        )

    def test_a_timer_only_soak_reads_and_checks_no_hook_evidence(self, checked: list[tuple[str, Any]]) -> None:
        """Hook checkers demand at least one hook fault, so running them on a timer soak would always fail."""
        self._check({_TIMER}, ft_components=("train", "rollout"), backend=ClusterBackend.RAY)

        assert checked == []

    @pytest.mark.parametrize(
        "ft_components,backend,expected",
        [
            (
                ("train", "rollout"),
                ClusterBackend.RAY,
                ["assert_hook_dispatches", "assert_p2p_receiver_failures", "assert_trainer_peers_progress"],
            ),
            (("rollout",), ClusterBackend.RAY, ["assert_hook_dispatches", "assert_p2p_receiver_failures"]),
            (("train",), ClusterBackend.RAY, ["assert_hook_dispatches", "assert_trainer_peers_progress"]),
            (
                ("train", "rollout"),
                ClusterBackend.KUBERNETES,
                ["assert_hook_dispatches", "assert_trainer_peers_progress"],
            ),
        ],
        ids=["ray-mixed", "ray-rollout", "ray-train", "k8s-mixed"],
    )
    def test_each_hook_checker_runs_only_where_its_faults_can_be_drawn(
        self,
        checked: list[tuple[str, Any]],
        ft_components: tuple[str, ...],
        backend: ClusterBackend,
        expected: list[str],
    ) -> None:
        """Receiver faults exist only for Ray rollout soaks and peer progress only for trainer soaks."""
        self._check({_TIMER, _HOOK}, ft_components=ft_components, backend=backend)

        assert checked[0] == ("read", "/dump")
        assert [name for name, _ in checked[1:]] == expected
        assert all(training_events == checked[1][1] for _, training_events in checked[1:])
