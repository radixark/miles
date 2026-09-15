import dataclasses
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import entrypoint, fault_forms, state
from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode
from tests.e2e.ft.conftest_ft.scenario_random_crash import _assert_drawn_fault_forms_worked, assert_healing

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

_ROLLOUT_CELL_NAME = "rollout-engine-0"
_ACTOR_CELL_NAME = "actor-0"


def _mode(*ft_components: str) -> FTTestMode:
    return dataclasses.replace(next(iter(MODES.values())), ft_components=tuple(ft_components))


def _injector(*, cell_type: str | None) -> entrypoint.FaultInjectorHandle:
    config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)
    return entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds=1e9,
        cell_type=cell_type,
        cell_fault_forms=fault_forms.create_cell_fault_forms(base_url="http://control", config=config),
    )


def _actor_cell(name: str = _ACTOR_CELL_NAME) -> dict:
    return {
        "metadata": {"name": name, "labels": {"miles.io/cell-type": "actor"}},
        "status": {"phase": "Running", "conditions": [{"type": "Healthy", "status": "True"}]},
    }


def _note_actor_injections(
    injector: entrypoint.FaultInjectorHandle, count: int, *, name: str = _ACTOR_CELL_NAME
) -> None:
    log = injector.event_log
    for _ in range(count):
        log.observe([_actor_cell(name)])
        log.note_injection_attempt(cell_name=name, form_name="inject_fault:sigkill", succeeded=True)


def _note_form_attempts(
    injector: entrypoint.FaultInjectorHandle, *, form_name: str, outcomes: list[bool], name: str = _ACTOR_CELL_NAME
) -> None:
    injector.event_log.observe([_actor_cell(name)])
    for succeeded in outcomes:
        injector.event_log.note_injection_attempt(cell_name=name, form_name=form_name, succeeded=succeeded)


def _note_rollout_injection(log: state.EventLog) -> None:
    log.note_injection_attempt(cell_name=_ROLLOUT_CELL_NAME, form_name="inject_fault:sigkill", succeeded=True)


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
        "metadata": {"name": _ROLLOUT_CELL_NAME, "labels": {"miles.io/cell-type": "rollout"}},
        "status": {"phase": phase, "conditions": conditions},
    }


def _write_shrink_only_events(event_dir: Path) -> None:
    event_logger = EventLogger(log_dir=event_dir, source=MainProcessIdentity())
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
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_healing(_mode("train"), injector=injector, dump_dir=str(tmp_path))

    def test_trainer_soak_ignores_rollout_injections_when_counting_its_own(self, tmp_path: Path) -> None:
        """A mixed soak's engine crashes say nothing about trainer healing, so they must not be counted."""
        _write_shrink_only_events(tmp_path / "events")
        injector = _injector(cell_type=None)
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        for _ in range(3):
            _note_rollout_injection(log)

        with pytest.raises(AssertionError, match="Soak proved too little"):
            assert_healing(_mode("train", "rollout"), injector=injector, dump_dir=str(tmp_path))

    def test_rollout_soak_rejects_unfinished_engine_recovery(self, tmp_path: Path) -> None:
        """A rollout-only soak that ends with an accepted injection still relaunching must fail."""
        injector = _injector(cell_type="rollout")
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])

        with pytest.raises(AssertionError, match="Rollout recovery witness failed"):
            assert_healing(_mode("rollout"), injector=injector, dump_dir=str(tmp_path))


def test_a_trainer_only_soak_targets_actor_cells() -> None:
    """It must not crash engines that its assertions say nothing about."""
    from tests.e2e.ft.conftest_ft.scenario_random_crash import compute_injected_cell_type

    assert compute_injected_cell_type(_mode("train")) == "actor"


def test_a_rollout_only_soak_targets_rollout_cells() -> None:
    """Crashing trainer cells here would exercise a component this mode did not enable ft on."""
    from tests.e2e.ft.conftest_ft.scenario_random_crash import compute_injected_cell_type

    assert compute_injected_cell_type(_mode("rollout")) == "rollout"


def test_a_mixed_soak_targets_every_kind() -> None:
    """The point of the mixed mode is that both kinds fail during one run."""
    from tests.e2e.ft.conftest_ft.scenario_random_crash import compute_injected_cell_type

    assert compute_injected_cell_type(_mode("train", "rollout")) is None


class TestAssertEveryDrawnFaultFormWorked:
    def test_a_form_that_never_worked_fails_the_soak(self, tmp_path: Path) -> None:
        """Pod deletion can be refused for the whole run while the kills alone clear the injection floor."""
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 3)
        _note_form_attempts(injector, form_name=fault_forms.DELETE_POD_FORM_NAME, outcomes=[False] * 4)

        with pytest.raises(AssertionError, match=fault_forms.DELETE_POD_FORM_NAME):
            _assert_drawn_fault_forms_worked(injector)

    def test_a_form_that_worked_at_least_once_is_accepted(self) -> None:
        """A single refusal is a cluster hiccup, not proof the fault form is wired up wrong."""
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 3)
        _note_form_attempts(injector, form_name=fault_forms.DELETE_POD_FORM_NAME, outcomes=[False, False, False, True])

        _assert_drawn_fault_forms_worked(injector)


class TestTrainerHealingPairing:
    def test_a_final_injection_that_never_healed_fails_even_though_the_floor_is_cleared(self, tmp_path: Path) -> None:
        """Regression: 3 crashes with 2 heals used to pass, leaving the run permanently degraded."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(_mode("train"), injector=injector, dump_dir=str(tmp_path))

    def test_two_cells_healed_by_one_reconfigure_event_count_as_two_healings(self, tmp_path: Path) -> None:
        """One reconfigure can readmit several cells, so counting events would under-count the healing."""
        _write_healing_events(tmp_path / "events", [[0, 1]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(_mode("train"), injector=injector, dump_dir=str(tmp_path))

    def test_healing_a_cell_that_was_never_injected_does_not_pay_another_cells_debt(self, tmp_path: Path) -> None:
        """Counting healings without pairing them by cell index would call this a healthy soak."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 2, name="actor-1")

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(_mode("train"), injector=injector, dump_dir=str(tmp_path))

    def test_every_injection_paired_with_a_healing_of_the_same_cell_passes(self, tmp_path: Path) -> None:
        """The assertion must stay invisible on the path a healthy soak actually takes."""
        _write_healing_events(tmp_path / "events", [[0], [1]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(_mode("train"), injector=injector, dump_dir=str(tmp_path))
