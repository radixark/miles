import dataclasses
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import entrypoint, state
from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode
from tests.e2e.ft.conftest_ft.scenario_random_crash import assert_healing

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity

_ROLLOUT_CELL_NAME = "rollout-engine-0"


def _mode(*ft_components: str) -> FTTestMode:
    return dataclasses.replace(next(iter(MODES.values())), ft_components=tuple(ft_components))


def _injector(*, cell_type: str, num_successful_injections: int) -> entrypoint.FaultInjectorHandle:
    injector = entrypoint.FaultInjectorHandle(
        base_url="http://control", seed=0, mean_interval_seconds=1e9, cell_type=cell_type
    )
    injector.num_successful_injections = num_successful_injections
    return injector


def _rollout_cell(state: state.ObservedCellState) -> dict:
    phase = "Pending" if state is state.ObservedCellState.PENDING else "Running"
    conditions = (
        []
        if phase == "Pending"
        else [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if state is state.ObservedCellState.SERVING else "False"},
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


class TestAssertHealing:
    def test_trainer_soak_rejects_missing_reconfigure_witness(self, tmp_path: Path) -> None:
        """A trainer-only soak whose accepted injections produced no healing event must fail."""
        _write_shrink_only_events(tmp_path / "events")

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_healing(
                _mode("train"),
                injector=_injector(cell_type="actor", num_successful_injections=3),
                dump_dir=str(tmp_path),
            )

    def test_rollout_soak_rejects_unfinished_engine_recovery(self, tmp_path: Path) -> None:
        """A rollout-only soak that ends with an accepted injection still relaunching must fail."""
        injector = _injector(cell_type="rollout", num_successful_injections=2)
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        log.note_injected(_ROLLOUT_CELL_NAME)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        log.note_injected(_ROLLOUT_CELL_NAME)
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
