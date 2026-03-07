"""Tests for StateMachineStepper and StateMachine base classes."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


# -- Dummy states for testing --------------------------------------------------


class DummyState(BaseModel):
    model_config = ConfigDict(frozen=True)


class StateA(DummyState):
    pass


class StateB(DummyState):
    value: int = 0


class StateC(DummyState):
    pass


class TerminalState(DummyState):
    pass


# -- Dummy stepper --------------------------------------------------------------


class DummyStepper(StateMachineStepper[DummyState, None]):
    def _build_handlers(self) -> dict:
        return {
            StateA: self._handle_a,
            StateB: self._handle_b,
            StateC: self._handle_c,
        }

    async def _handle_a(self, state: StateA, _context: None) -> DummyState:
        return StateB(value=1)

    async def _handle_b(self, state: StateB, _context: None) -> DummyState | None:
        if state.value >= 3:
            return TerminalState()
        return StateB(value=state.value + 1)

    async def _handle_c(self, state: StateC, _context: None) -> DummyState | None:
        return None


# -- Tests: StateMachineStepper ------------------------------------------------


class TestStateMachineStepper:
    @pytest.mark.asyncio
    async def test_dispatch_to_correct_handler(self) -> None:
        stepper = DummyStepper()
        result = await stepper(StateA(), None)
        assert isinstance(result, StateB)
        assert result.value == 1

    @pytest.mark.asyncio
    async def test_terminal_state_returns_none(self) -> None:
        stepper = DummyStepper()
        result = await stepper(TerminalState(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_handler_returning_none(self) -> None:
        stepper = DummyStepper()
        result = await stepper(StateC(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_same_type_transition(self) -> None:
        stepper = DummyStepper()
        result = await stepper(StateB(value=1), None)
        assert isinstance(result, StateB)
        assert result.value == 2

    @pytest.mark.asyncio
    async def test_transition_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = DummyStepper()
        with caplog.at_level("INFO"):
            await stepper(StateA(), None)
        assert "DummyStepper StateA -> StateB" in caplog.text


# -- Tests: StateMachine -------------------------------------------------------


class TestStateMachine:
    @pytest.mark.asyncio
    async def test_step_runs_until_none(self) -> None:
        """StateA -> StateB(1) -> StateB(2) -> StateB(3) -> TerminalState -> (stepper returns None)."""
        machine = StateMachine(initial_state=StateA(), stepper=DummyStepper())
        await machine.step(None)

        assert isinstance(machine.state, TerminalState)
        assert len(machine.state_history) == 4

    @pytest.mark.asyncio
    async def test_step_no_transition(self) -> None:
        machine = StateMachine(initial_state=StateC(), stepper=DummyStepper())
        await machine.step(None)
        assert isinstance(machine.state, StateC)
        assert len(machine.state_history) == 0

    @pytest.mark.asyncio
    async def test_step_already_terminal(self) -> None:
        machine = StateMachine(initial_state=TerminalState(), stepper=DummyStepper())
        await machine.step(None)
        assert isinstance(machine.state, TerminalState)
        assert len(machine.state_history) == 0

    @pytest.mark.asyncio
    async def test_state_history_records_all_transitions(self) -> None:
        machine = StateMachine(initial_state=StateA(), stepper=DummyStepper())
        await machine.step(None)

        types = [type(s).__name__ for s in machine.state_history]
        assert types == ["StateB", "StateB", "StateB", "TerminalState"]

    @pytest.mark.asyncio
    async def test_stepper_property(self) -> None:
        stepper = DummyStepper()
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        assert machine.stepper is stepper
