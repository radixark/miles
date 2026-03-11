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


class UnregisteredState(DummyState):
    pass


# -- Dummy handlers ------------------------------------------------------------


class StateAHandler:
    async def step(self, state: StateA, _context: None) -> DummyState:
        return StateB(value=1)


class StateBHandler:
    async def step(self, state: StateB, _context: None) -> DummyState | None:
        if state.value >= 3:
            return TerminalState()
        return StateB(value=state.value + 1)


class StateCHandler:
    async def step(self, state: StateC, _context: None) -> DummyState | None:
        return None


class TerminalStateHandler:
    async def step(self, state: TerminalState, _context: None) -> DummyState | None:
        return None


class StateAGenHandler:
    """Yields multiple states (async generator handler)."""

    async def step(self, state: StateA, _context: None):
        yield StateB(value=1)
        yield StateB(value=2)


class EmptyGenHandler:
    """Yields nothing (async generator that returns immediately)."""

    async def step(self, state: StateA, _context: None):
        return


class SingleYieldGenHandler:
    """Yields exactly one state."""

    async def step(self, state: StateA, _context: None):
        yield StateB(value=42)


class SameStateGenHandler:
    """Yields a state identical to the input."""

    async def step(self, state: StateA, _context: None):
        yield StateA()


HANDLER_MAP: dict[type, type] = {
    StateA: StateAHandler,
    StateB: StateBHandler,
    StateC: StateCHandler,
    TerminalState: TerminalStateHandler,
}


def _make_stepper(**kwargs) -> StateMachineStepper[DummyState, None]:
    return StateMachineStepper(handler_map=HANDLER_MAP, **kwargs)


def _make_gen_stepper(
    handler_cls: type = StateAGenHandler,
    **kwargs,
) -> StateMachineStepper[DummyState, None]:
    handler_map: dict[type, type] = {
        StateA: handler_cls,
        StateB: StateBHandler,
        TerminalState: TerminalStateHandler,
    }
    return StateMachineStepper(handler_map=handler_map, **kwargs)


async def _step_last(stepper: StateMachineStepper, state, ctx):
    result = None
    async for result in stepper(state, ctx):
        pass
    return result


# -- Tests: StateMachineStepper ------------------------------------------------


class TestStateMachineStepper:
    @pytest.mark.asyncio
    async def test_dispatch_to_correct_handler(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper, StateA(), None)
        assert isinstance(result, StateB)
        assert result.value == 1

    @pytest.mark.asyncio
    async def test_terminal_state_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper, TerminalState(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_handler_returning_none(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper, StateC(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_unregistered_state_raises_type_error(self) -> None:
        stepper = _make_stepper()
        with pytest.raises(TypeError, match="has no handler for state type UnregisteredState"):
            await _step_last(stepper, UnregisteredState(), None)

    @pytest.mark.asyncio
    async def test_same_type_transition(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper, StateB(value=1), None)
        assert isinstance(result, StateB)
        assert result.value == 2

    @pytest.mark.asyncio
    async def test_stepper_does_not_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Logging moved to StateMachine — stepper should not log."""
        stepper = _make_stepper()
        with caplog.at_level("INFO"):
            await _step_last(stepper, StateA(), None)
        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_pre_dispatch_short_circuits(self) -> None:
        """pre_dispatch returning non-None skips handler dispatch."""

        async def always_terminal(state: DummyState, ctx: None) -> DummyState | None:
            return TerminalState()

        stepper = _make_stepper(pre_dispatch=always_terminal)
        result = await _step_last(stepper, StateA(), None)
        assert isinstance(result, TerminalState)

    @pytest.mark.asyncio
    async def test_pre_dispatch_none_falls_through(self) -> None:
        """pre_dispatch returning None continues to normal handler dispatch."""

        async def pass_through(state: DummyState, ctx: None) -> DummyState | None:
            return None

        stepper = _make_stepper(pre_dispatch=pass_through)
        result = await _step_last(stepper, StateA(), None)
        assert isinstance(result, StateB)
        assert result.value == 1


# -- Tests: StateMachine -------------------------------------------------------


class TestStateMachine:
    @pytest.mark.asyncio
    async def test_step_runs_until_none(self) -> None:
        """StateA -> StateB(1) -> StateB(2) -> StateB(3) -> TerminalState -> (stepper returns None)."""
        machine = StateMachine(initial_state=StateA(), stepper=_make_stepper())
        await machine.step(None)

        assert isinstance(machine.state, TerminalState)
        assert len(machine.state_history) == 4

    @pytest.mark.asyncio
    async def test_step_no_transition(self) -> None:
        machine = StateMachine(initial_state=StateC(), stepper=_make_stepper())
        await machine.step(None)
        assert isinstance(machine.state, StateC)
        assert len(machine.state_history) == 0

    @pytest.mark.asyncio
    async def test_step_already_terminal(self) -> None:
        machine = StateMachine(initial_state=TerminalState(), stepper=_make_stepper())
        await machine.step(None)
        assert isinstance(machine.state, TerminalState)
        assert len(machine.state_history) == 0

    @pytest.mark.asyncio
    async def test_state_history_records_all_transitions(self) -> None:
        machine = StateMachine(initial_state=StateA(), stepper=_make_stepper())
        await machine.step(None)

        types = [type(s).__name__ for s in machine.state_history]
        assert types == ["StateB", "StateB", "StateB", "TerminalState"]

    @pytest.mark.asyncio
    async def test_stepper_property(self) -> None:
        stepper = _make_stepper()
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        assert machine.stepper is stepper


# -- Tests: StateMachineStepper — generator support ----------------------------


class TestStateMachineStepperGenerator:
    @pytest.mark.asyncio
    async def test_gen_handler_yields_states(self) -> None:
        stepper = _make_gen_stepper(StateAGenHandler)
        results = [s async for s in stepper(StateA(), None)]
        assert results == [StateB(value=1), StateB(value=2)]

    @pytest.mark.asyncio
    async def test_gen_handler_step_last_returns_last(self) -> None:
        stepper = _make_gen_stepper(StateAGenHandler)
        result = await _step_last(stepper, StateA(), None)
        assert result == StateB(value=2)

    @pytest.mark.asyncio
    async def test_empty_gen_yields_nothing(self) -> None:
        stepper = _make_gen_stepper(EmptyGenHandler)
        results = [s async for s in stepper(StateA(), None)]
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_gen_step_last_returns_none(self) -> None:
        stepper = _make_gen_stepper(EmptyGenHandler)
        result = await _step_last(stepper, StateA(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_single_yield_gen(self) -> None:
        stepper = _make_gen_stepper(SingleYieldGenHandler)
        results = [s async for s in stepper(StateA(), None)]
        assert results == [StateB(value=42)]

    @pytest.mark.asyncio
    async def test_single_yield_gen_step_last(self) -> None:
        stepper = _make_gen_stepper(SingleYieldGenHandler)
        result = await _step_last(stepper, StateA(), None)
        assert result == StateB(value=42)

    @pytest.mark.asyncio
    async def test_gen_handler_terminal_state_skips(self) -> None:
        stepper = _make_gen_stepper(
            StateAGenHandler,
            terminal_states=frozenset({StateA}),
        )
        results = [s async for s in stepper(StateA(), None)]
        assert results == []

    @pytest.mark.asyncio
    async def test_gen_handler_pre_dispatch_short_circuits(self) -> None:
        async def always_terminal(state: DummyState, ctx: None) -> DummyState | None:
            return TerminalState()

        stepper = _make_gen_stepper(StateAGenHandler, pre_dispatch=always_terminal)
        results = [s async for s in stepper(StateA(), None)]
        assert results == [TerminalState()]

    @pytest.mark.asyncio
    async def test_gen_handler_unregistered_raises(self) -> None:
        stepper = _make_gen_stepper(StateAGenHandler)
        with pytest.raises(TypeError, match="has no handler for state type"):
            await _step_last(stepper, UnregisteredState(), None)


# -- Tests: _step_last regression -----------------------------------------------


class TestStepLastRegression:
    @pytest.mark.asyncio
    async def test_step_last_regular_handler(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper, StateA(), None)
        assert result == StateB(value=1)

    @pytest.mark.asyncio
    async def test_step_last_returns_none_on_no_transition(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper, StateC(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_step_last_terminal_state(self) -> None:
        stepper = _make_stepper(terminal_states=frozenset({TerminalState}))
        result = await _step_last(stepper, TerminalState(), None)
        assert result is None


# -- Tests: StateMachine — generator & history/logging integration -------------


class TestStateMachineGenerator:
    @pytest.mark.asyncio
    async def test_gen_handler_all_yields_in_history(self) -> None:
        """Gen handler yields StateB(1), StateB(2) → then StateBHandler chains to TerminalState."""
        stepper = _make_gen_stepper(StateAGenHandler)
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        await machine.step(None)

        assert machine.state == TerminalState()
        values = [s.value for s in machine.state_history if isinstance(s, StateB)]
        assert 1 in values
        assert 2 in values

    @pytest.mark.asyncio
    async def test_gen_handler_empty_no_transition(self) -> None:
        stepper = _make_gen_stepper(EmptyGenHandler)
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        await machine.step(None)

        assert machine.state == StateA()
        assert len(machine.state_history) == 0

    @pytest.mark.asyncio
    async def test_gen_handler_logging_per_transition(self, caplog: pytest.LogCaptureFixture) -> None:
        """Each state change from generator produces an INFO log."""
        stepper = _make_gen_stepper(StateAGenHandler)
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        with caplog.at_level("INFO"):
            await machine.step(None)

        assert "StateA()" in caplog.text
        assert "StateB(value=1)" in caplog.text
        assert "StateB(value=2)" in caplog.text

    @pytest.mark.asyncio
    async def test_same_state_yield_no_transition_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """When a gen handler yields the same state, no '-> ' log is produced."""
        stepper = _make_gen_stepper(SameStateGenHandler)

        with caplog.at_level("INFO", logger="miles.utils.ft.utils.state_machine"):
            results = [s async for s in stepper(StateA(), None)]

        assert results == [StateA()]
        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_gen_then_regular_handler_chain(self) -> None:
        """Gen handler yields StateB(1) → regular StateBHandler chains: B(2) → B(3) → TerminalState."""
        stepper = _make_gen_stepper(SingleYieldGenHandler)
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        await machine.step(None)

        assert machine.state == TerminalState()
        types = [type(s).__name__ for s in machine.state_history]
        assert "StateB" in types
        assert "TerminalState" in types

    @pytest.mark.asyncio
    async def test_mixed_gen_and_regular_handlers(self) -> None:
        """StateA (gen handler) yields StateB → StateBHandler (regular) continues chain."""
        stepper = _make_gen_stepper(StateAGenHandler)
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        await machine.step(None)

        assert isinstance(machine.state, TerminalState)
        history_types = [type(s).__name__ for s in machine.state_history]
        assert "StateB" in history_types
        assert "TerminalState" in history_types
