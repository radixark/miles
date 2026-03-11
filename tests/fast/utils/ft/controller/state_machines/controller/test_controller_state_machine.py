"""Tests for the controller state machine (ControllerState + handlers)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.ft.utils.controller_fakes import FakeMainJob

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.controller import (
    ControllerContext,
    NormalState,
    RestartingMainJobState,
    create_controller_stepper,
)
from miles.utils.ft.controller.state_machines.main.models import (
    DetectingAnomaly,
    RestartedMainJob,
    RestartingMainJob,
)
from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_stepper() -> StateMachineStepper:
    """Stepper with no handlers — step() is a no-op for any state."""
    return StateMachineStepper(
        handler_map={
            DetectingAnomaly: type("_NoopHandler", (), {"step": AsyncMock(return_value=None)}),
        },
        terminal_states=frozenset({RestartingMainJob, RestartedMainJob}),
    )


def _make_sub_sm(initial_state=None) -> StateMachine:
    return StateMachine(
        initial_state=initial_state or DetectingAnomaly(),
        stepper=_noop_stepper(),
    )


def _make_subsystem(name: str, state_machine: StateMachine | None = None) -> SubsystemEntry:
    return SubsystemEntry(
        name=name,
        state_machine=state_machine or _make_sub_sm(),
        actuator=AsyncMock(),
    )


def _make_controller_context(
    *,
    main_job: FakeMainJob | None = None,
    fresh_subsystems: dict[str, SubsystemEntry] | None = None,
) -> ControllerContext:
    return ControllerContext(
        main_job=main_job or FakeMainJob(),
        create_fresh_subsystems=lambda: fresh_subsystems or {
            "training": _make_subsystem("training"),
        },
    )


# ---------------------------------------------------------------------------
# Scenario 1: Normal operation — no state transition
# ---------------------------------------------------------------------------


class TestNormalOperation:
    @pytest.mark.asyncio
    async def test_all_subsystems_detecting_no_transition(self) -> None:
        """Two sub-SMs in DetectingAnomaly → NormalStateHandler steps them → no transition."""
        stepper = create_controller_stepper()
        subsystems = {
            "training": _make_subsystem("training"),
            "rollout_0": _make_subsystem("rollout_0"),
        }
        state = NormalState(subsystems=subsystems)
        context = _make_controller_context()

        result = await stepper(state, context)

        assert result is None


# ---------------------------------------------------------------------------
# Scenario 2: Single subsystem escalation
# ---------------------------------------------------------------------------


class TestSingleSubsystemEscalation:
    @pytest.mark.asyncio
    async def test_sub_sm_in_restarting_main_job_triggers_escalation(self) -> None:
        """One sub-SM in RestartingMainJob → stop+submit main job → RestartingMainJobState."""
        main_job = FakeMainJob()
        stepper = create_controller_stepper()

        sub_sm = _make_sub_sm()
        sub_sm.force_state(RestartingMainJob())

        subsystems = {
            "rollout_0": _make_subsystem("rollout_0", state_machine=sub_sm),
        }
        state = NormalState(subsystems=subsystems)
        context = _make_controller_context(main_job=main_job)

        result = await stepper(state, context)

        assert isinstance(result, RestartingMainJobState)
        assert result.requestor_name == "rollout_0"
        assert main_job._stopped
        assert main_job._submitted


# ---------------------------------------------------------------------------
# Scenario 3: Job restart complete → reset to NormalState
# ---------------------------------------------------------------------------


class TestJobRestartComplete:
    @pytest.mark.asyncio
    async def test_running_status_rebuilds_subsystems_and_signals_requestor(self) -> None:
        """RestartingMainJobState + RUNNING → create_fresh_subsystems → requestor force_state(RestartedMainJob)."""
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
        stepper = create_controller_stepper()

        fresh_training_sm = _make_sub_sm()
        fresh_rollout_sm = _make_sub_sm()
        fresh_subsystems = {
            "training": _make_subsystem("training", state_machine=fresh_training_sm),
            "rollout_0": _make_subsystem("rollout_0", state_machine=fresh_rollout_sm),
        }

        state = RestartingMainJobState(requestor_name="rollout_0")
        context = _make_controller_context(
            main_job=main_job,
            fresh_subsystems=fresh_subsystems,
        )

        result = await stepper(state, context)

        assert isinstance(result, NormalState)
        assert set(result.subsystems.keys()) == {"training", "rollout_0"}
        assert isinstance(
            result.subsystems["rollout_0"].state_machine.state,
            RestartedMainJob,
        )
        assert isinstance(
            result.subsystems["training"].state_machine.state,
            DetectingAnomaly,
        )


# ---------------------------------------------------------------------------
# Scenario 4: Multiple subsystems, one escalates
# ---------------------------------------------------------------------------


class TestMultiSubsystemOneEscalates:
    @pytest.mark.asyncio
    async def test_only_one_escalating_triggers_single_restart(self) -> None:
        """Two subsystems, only rollout_0 in RestartingMainJob → one job restart, both rebuilt."""
        main_job = FakeMainJob()
        stepper = create_controller_stepper()

        training_sm = _make_sub_sm()
        rollout_sm = _make_sub_sm()
        rollout_sm.force_state(RestartingMainJob())

        subsystems = {
            "training": _make_subsystem("training", state_machine=training_sm),
            "rollout_0": _make_subsystem("rollout_0", state_machine=rollout_sm),
        }
        state = NormalState(subsystems=subsystems)

        fresh_subsystems = {
            "training": _make_subsystem("training"),
            "rollout_0": _make_subsystem("rollout_0"),
        }
        context = _make_controller_context(
            main_job=main_job,
            fresh_subsystems=fresh_subsystems,
        )

        result = await stepper(state, context)

        assert isinstance(result, RestartingMainJobState)
        assert result.requestor_name == "rollout_0"
        assert main_job._submit_call_count == 1


# ---------------------------------------------------------------------------
# Scenario 5: Job restart pending — keep waiting
# ---------------------------------------------------------------------------


class TestJobRestartPending:
    @pytest.mark.asyncio
    async def test_pending_status_returns_none(self) -> None:
        """RestartingMainJobState + PENDING → handler returns None, stay waiting."""
        main_job = FakeMainJob(status_sequence=[JobStatus.PENDING])
        stepper = create_controller_stepper()

        state = RestartingMainJobState(requestor_name="rollout_0")
        context = _make_controller_context(main_job=main_job)

        result = await stepper(state, context)

        assert result is None

    @pytest.mark.asyncio
    async def test_failed_status_rebuilds_for_retry(self) -> None:
        """RestartingMainJobState + FAILED → rebuild subsystems for retry."""
        main_job = FakeMainJob(status_sequence=[JobStatus.FAILED])
        stepper = create_controller_stepper()

        state = RestartingMainJobState(requestor_name="rollout_0")
        context = _make_controller_context(main_job=main_job)

        result = await stepper(state, context)

        assert isinstance(result, NormalState)
