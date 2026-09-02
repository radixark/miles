from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import entrypoint, fault_forms, state, views
from tests.e2e.ft.conftest_ft.scenario_random_crash import (
    DEFAULT_NUM_STEPS,
    TERMINAL_FAULT_FREE_STEPS,
    _assert_every_drawn_fault_form_worked,
    _fault_injection_enabled,
    assert_healing,
)

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

_ROLLOUT_CELL_NAME = "rollout-engine-0"
_ACTOR_CELL_NAME = "actor-0"


def _injector(*, cell_types: tuple[str, ...]) -> entrypoint.FaultInjectorHandle:
    config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)
    return entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type={cell_type: 1e9 for cell_type in cell_types},
        cell_fault_forms=fault_forms.create_cell_fault_forms(base_url="http://control", config=config),
    )


def _actor_cell(name: str = _ACTOR_CELL_NAME) -> dict:
    return {
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": "actor", "miles.io/workers-hash": "generation-0"},
        },
        "status": {"phase": "Running", "conditions": [{"type": "Healthy", "status": "True"}]},
    }


def _note_actor_injections(
    injector: entrypoint.FaultInjectorHandle, count: int, *, name: str = _ACTOR_CELL_NAME
) -> None:
    log = injector.event_log
    for _ in range(count):
        log.observe([_actor_cell(name)])
        log.note_injection_attempt(
            cell_name=name,
            form_name="inject_fault:sigkill",
            succeeded=True,
        )


def _note_form_attempts(
    injector: entrypoint.FaultInjectorHandle, *, form_name: str, outcomes: list[bool], name: str = _ACTOR_CELL_NAME
) -> None:
    injector.event_log.observe([_actor_cell(name)])
    for succeeded in outcomes:
        injector.event_log.note_injection_attempt(
            cell_name=name,
            form_name=form_name,
            succeeded=succeeded,
        )


def _note_rollout_injection(log: state.EventLog) -> None:
    log.note_injection_attempt(
        cell_name=_ROLLOUT_CELL_NAME,
        form_name="inject_fault:sigkill",
        succeeded=True,
    )


def _rollout_cell(cell_state: state.ObservedCellState) -> dict:
    phase = "Pending" if cell_state is state.ObservedCellState.PENDING else "Running"
    conditions = (
        []
        if phase == "Pending"
        else [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if cell_state is state.ObservedCellState.SERVING else "False"},
        ]
    )
    return {
        "metadata": {
            "name": _ROLLOUT_CELL_NAME,
            "labels": {"miles.io/cell-type": "rollout", "miles.io/workers-hash": "generation-0"},
        },
        "status": {"phase": phase, "conditions": conditions},
    }


def _write_shrink_only_events(event_dir: Path) -> None:
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="main"))
    event_logger.log(
        CellReconfigureEvent,
        dict(rollout_id=2, quorum_id=1, src_cell_index=None, healed_cell_indices=[], alive_cell_indices_after=[0]),
        print_log=False,
    )
    event_logger.close()


def _write_healing_events(event_dir: Path, healed_cell_indices_per_event: list[list[int]]) -> None:
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="main"))
    for index, healed_cell_indices in enumerate(healed_cell_indices_per_event):
        event_logger.log(
            CellReconfigureEvent,
            dict(
                rollout_id=index + 2,
                quorum_id=index + 1,
                src_cell_index=0,
                healed_cell_indices=healed_cell_indices,
                alive_cell_indices_after=[0, 1],
            ),
            print_log=False,
        )
    event_logger.close()


class TestAssertHealing:
    def test_trainer_soak_rejects_missing_reconfigure_witness(self, tmp_path: Path) -> None:
        """A trainer-only soak whose accepted injections produced no healing event must fail."""
        _write_shrink_only_events(tmp_path / "events")
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_trainer_soak_ignores_rollout_injections_when_counting_its_own(self, tmp_path: Path) -> None:
        """A mixed soak's engine crashes say nothing about trainer healing, so they must not be counted."""
        _write_shrink_only_events(tmp_path / "events")
        injector = _injector(cell_types=("actor", "rollout"))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        for _ in range(3):
            _note_rollout_injection(log)

        with pytest.raises(AssertionError, match="Soak proved too little"):
            assert_healing(("train", "rollout"), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_rollout_soak_rejects_a_cell_never_seen_serving_after_its_last_injection(self, tmp_path: Path) -> None:
        """A rollout-only soak that ends with its last victim still relaunching must fail."""
        injector = _injector(cell_types=("rollout",))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])

        with pytest.raises(AssertionError, match="Rollout recovery witness failed"):
            assert_healing(("rollout",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_rollout_soak_accepts_a_fresh_serve_after_the_last_injection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The witness must stay invisible on the path a healthy soak actually takes."""
        monkeypatch.setattr(views, "STALE_STATUS_GRACE_SECONDS", 0.0)
        injector = _injector(cell_types=("rollout",))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        for _ in range(2):
            _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])

        assert_healing(("rollout",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_rollout_soak_rejects_a_serve_still_inside_the_stale_window(self, tmp_path: Path) -> None:
        """A serve observed right after the kill can be the dead cell's stale reading, and proves nothing."""
        injector = _injector(cell_types=("rollout",))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        for _ in range(2):
            _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])

        with pytest.raises(AssertionError, match="Rollout recovery witness failed"):
            assert_healing(("rollout",), injector=injector, event_dir=tmp_path / "events", context="soak")


def _mean_intervals(*ft_components: str) -> dict[str, float]:
    return fault_forms.compute_mean_interval_seconds_of_cell_type(
        tuple(ft_components), trainer_crash_interval_seconds=120.0, rollout_crash_interval_seconds=240.0
    )


def test_a_trainer_only_soak_schedules_actor_injections_only() -> None:
    """It must not crash engines that its assertions say nothing about."""
    assert _mean_intervals("train") == {"actor": 120.0}


def test_a_rollout_only_soak_schedules_rollout_injections_only() -> None:
    """Crashing trainer cells here would exercise a component this mode did not enable ft on."""
    assert _mean_intervals("rollout") == {"rollout": 240.0}


def test_a_mixed_soak_keeps_each_kind_on_the_cadence_it_would_have_alone() -> None:
    """Adding rollout to a soak must not dilute the trainer crash rate it was calibrated at."""
    assert _mean_intervals("train", "rollout") == {"actor": 120.0, "rollout": 240.0}


def test_a_kind_the_mode_does_not_enable_ft_on_gets_no_schedule_at_all() -> None:
    """An entry in the map is what makes the loop consider a kind, so a stray one crashes an unwatched component."""
    assert "rollout" not in _mean_intervals("train")


class TestAssertEveryDrawnFaultFormWorked:
    def test_a_form_that_never_worked_fails_the_soak(self, tmp_path: Path) -> None:
        """Pod deletion can be refused for the whole run while the kills alone clear the injection floor."""
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)
        _note_form_attempts(injector, form_name=fault_forms.DELETE_POD_FORM_NAME, outcomes=[False] * 4)

        with pytest.raises(AssertionError, match=fault_forms.DELETE_POD_FORM_NAME):
            _assert_every_drawn_fault_form_worked(injector)

    def test_a_form_that_worked_at_least_once_is_accepted(self) -> None:
        """A single refusal is a cluster hiccup, not proof the fault form is wired up wrong."""
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)
        _note_form_attempts(injector, form_name=fault_forms.DELETE_POD_FORM_NAME, outcomes=[False, False, False, True])

        _assert_every_drawn_fault_form_worked(injector)


class TestTrainerHealingPairing:
    def test_a_final_injection_that_never_healed_fails_even_though_the_floor_is_cleared(self, tmp_path: Path) -> None:
        """Regression: 3 crashes with 2 heals used to pass, leaving the run permanently degraded."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_two_cells_healed_by_one_reconfigure_event_count_as_two_healings(self, tmp_path: Path) -> None:
        """One reconfigure can readmit several cells, so counting events would under-count the healing."""
        _write_healing_events(tmp_path / "events", [[0, 1]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_healing_a_cell_that_was_never_injected_does_not_pay_another_cells_debt(self, tmp_path: Path) -> None:
        """Counting healings without pairing them by cell index would call this a healthy soak."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 2, name="actor-1")

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_every_injection_paired_with_a_healing_of_the_same_cell_passes(self, tmp_path: Path) -> None:
        """The assertion must stay invisible on the path a healthy soak actually takes."""
        _write_healing_events(tmp_path / "events", [[0], [1]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")


class TestTerminalFaultFreeTail:
    def test_the_fault_window_closes_before_the_final_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The soak's last steps admit no new fault, so the last accepted one still has steps to heal in."""
        _stub_completed_rollouts(monkeypatch, [DEFAULT_NUM_STEPS - TERMINAL_FAULT_FREE_STEPS - 1])

        assert not _fault_injection_enabled("/dump", num_steps=DEFAULT_NUM_STEPS)

    def test_the_fault_window_is_open_one_step_earlier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tail that swallowed the whole soak would make the witnesses unreachable instead of reliable."""
        _stub_completed_rollouts(monkeypatch, [DEFAULT_NUM_STEPS - TERMINAL_FAULT_FREE_STEPS - 2])

        assert _fault_injection_enabled("/dump", num_steps=DEFAULT_NUM_STEPS)

    def test_a_soak_that_has_finished_no_rollout_yet_may_be_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The very first fault must not wait for a completed rollout that a crash may be delaying."""
        _stub_completed_rollouts(monkeypatch, [])

        assert _fault_injection_enabled("/dump", num_steps=DEFAULT_NUM_STEPS)

    def test_the_watermark_is_the_latest_rollout_id_not_the_event_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retried rollouts emit several metric events, which must not close the window early."""
        _stub_completed_rollouts(monkeypatch, [0, 0, 0, 1, 1])

        assert _fault_injection_enabled("/dump", num_steps=DEFAULT_NUM_STEPS)


def _stub_completed_rollouts(monkeypatch: pytest.MonkeyPatch, rollout_ids: list[int]) -> None:
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    completions = [(rollout_id, base + timedelta(seconds=index)) for index, rollout_id in enumerate(rollout_ids)]
    monkeypatch.setattr(
        "tests.e2e.ft.conftest_ft.scenario_random_crash.read_rollout_completion_times",
        lambda dump_dir: completions,
    )
