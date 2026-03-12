"""Tests for StateMachineStepper, StateMachine, and run_stepper_to_convergence."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, ConfigDict

from miles.utils.ft.utils.state_machine import (
    StateMachine,
    StateMachineStepper,
    _to_async_gen,
    run_stepper_to_convergence,
)


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


class OscillatingAHandler:
    """A → B(0), creating an A-B-A-B... cycle that never converges."""

    async def step(self, state: StateA, _context: None) -> DummyState:
        return StateB(value=0)


class OscillatingBHandler:
    """B → A, creating an A-B-A-B... cycle that never converges."""

    async def step(self, state: StateB, _context: None) -> DummyState:
        return StateA()


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
    async def test_step_oscillating_terminates_at_max_iterations(self) -> None:
        """Previously StateMachine.step() used while True with no bound.
        Two oscillating handlers (A→B→A→…) would livelock the controller.
        Now step() is capped at _MAX_CONVERGENCE_ITERATIONS (same as run_stepper_to_convergence).
        """
        stepper = _make_oscillating_stepper()
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        await machine.step(None)
        assert len(machine.state_history) <= 50

    @pytest.mark.asyncio
    async def test_step_oscillating_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Max-iteration guard logs a warning on cap hit."""
        stepper = _make_oscillating_stepper()
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            await machine.step(None)
        assert "StateMachine.step hit max iterations" in caplog.text

    @pytest.mark.asyncio
    async def test_state_history_records_all_transitions(self) -> None:
        machine = StateMachine(initial_state=StateA(), stepper=_make_stepper())
        await machine.step(None)

        types = [type(s).__name__ for s in machine.state_history]
        assert types == ["StateB", "StateB", "StateB", "TerminalState"]


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


# -- Helpers for run_stepper_to_convergence tests ------------------------------


def _make_oscillating_stepper() -> StateMachineStepper[DummyState, None]:
    """A→B→A→B→... never converges."""
    return StateMachineStepper(handler_map={
        StateA: OscillatingAHandler,
        StateB: OscillatingBHandler,
    })


def _make_null_stepper() -> StateMachineStepper[DummyState, None]:
    """StateA always returns None — no transition."""

    class _NullHandler:
        async def step(self, state: StateA, _ctx: None) -> None:
            return None

    return StateMachineStepper(handler_map={StateA: _NullHandler})


# -- Tests: run_stepper_to_convergence -----------------------------------------


class TestRunStepperToConvergenceNoTransition:
    @pytest.mark.asyncio
    async def test_handler_returns_none_yields_nothing(self) -> None:
        results = [s async for s in run_stepper_to_convergence(_make_null_stepper(), StateA(), None)]
        assert results == []


class TestRunStepperToConvergenceChain:
    @pytest.mark.asyncio
    async def test_full_chain_yields_all_intermediates(self) -> None:
        """A→B(1)→B(2)→B(3)→Terminal: 4 stepper dispatches, 4 yields."""
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        assert len(results) == 4
        assert results[0] == StateB(value=1)
        assert results[1] == StateB(value=2)
        assert results[2] == StateB(value=3)
        assert results[3] == TerminalState()

    @pytest.mark.asyncio
    async def test_starting_mid_chain_converges(self) -> None:
        """B(2)→B(3)→Terminal: start from middle of chain."""
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateB(value=2), None)]

        assert results == [StateB(value=3), TerminalState()]

    @pytest.mark.asyncio
    async def test_starting_at_terminal_yields_nothing(self) -> None:
        stepper = _make_stepper(terminal_states=frozenset({TerminalState}))
        results = [s async for s in run_stepper_to_convergence(stepper, TerminalState(), None)]
        assert results == []

    @pytest.mark.asyncio
    async def test_yields_are_in_transition_order(self) -> None:
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        state_b_values = [s.value for s in results if isinstance(s, StateB)]
        assert state_b_values == sorted(state_b_values)


class TestRunStepperToConvergenceGenerator:
    @pytest.mark.asyncio
    async def test_gen_handler_yields_all_intermediate_states(self) -> None:
        """Gen handler yields B(1), B(2) in one step; then B handler chains to Terminal."""
        stepper = _make_gen_stepper(StateAGenHandler)
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        assert results[0] == StateB(value=1)
        assert results[1] == StateB(value=2)
        assert results[-1] == TerminalState()

    @pytest.mark.asyncio
    async def test_gen_handler_both_yields_visible(self) -> None:
        """Both yields from the gen handler appear — not just the last."""
        stepper = _make_gen_stepper(StateAGenHandler)
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        values = [s.value for s in results if isinstance(s, StateB)]
        assert 1 in values
        assert 2 in values


class TestRunStepperToConvergenceMaxIterations:
    @pytest.mark.asyncio
    async def test_oscillating_stops_at_max_iterations(self) -> None:
        """A→B→A→B→... should stop at max_iterations."""
        stepper = _make_oscillating_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None, max_iterations=5)]

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_max_iterations_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = _make_oscillating_stepper()
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            _ = [s async for s in run_stepper_to_convergence(stepper, StateA(), None, max_iterations=3)]

        assert "hit max iterations (3)" in caplog.text

    @pytest.mark.asyncio
    async def test_normal_convergence_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = _make_stepper()
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            _ = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        assert "hit max iterations" not in caplog.text

    @pytest.mark.asyncio
    async def test_max_iterations_one_yields_single_dispatch(self) -> None:
        """max_iterations=1 allows exactly one stepper dispatch."""
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None, max_iterations=1)]

        assert len(results) == 1
        assert results[0] == StateB(value=1)

    @pytest.mark.asyncio
    async def test_oscillating_yields_alternate_states(self) -> None:
        """Verify the oscillating pattern: A→B(0)→A→B(0)→A."""
        stepper = _make_oscillating_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None, max_iterations=4)]

        types = [type(s).__name__ for s in results]
        assert types == ["StateB", "StateA", "StateB", "StateA"]


# -- Tests: _to_async_gen -----------------------------------------------------


class TestToAsyncGen:
    @pytest.mark.asyncio
    async def test_coroutine_returning_value(self) -> None:
        async def coro() -> int:
            return 42

        results = [x async for x in _to_async_gen(coro())]
        assert results == [42]

    @pytest.mark.asyncio
    async def test_coroutine_returning_none(self) -> None:
        async def coro() -> None:
            return None

        results = [x async for x in _to_async_gen(coro())]
        assert results == [None]

    @pytest.mark.asyncio
    async def test_async_gen_multiple_items(self) -> None:
        async def gen():
            yield 1
            yield 2
            yield 3

        results = [x async for x in _to_async_gen(gen())]
        assert results == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_async_gen_with_none_items(self) -> None:
        """None values are yielded through — filtering is the caller's responsibility."""
        async def gen():
            yield "a"
            yield None
            yield "b"

        results = [x async for x in _to_async_gen(gen())]
        assert results == ["a", None, "b"]

    @pytest.mark.asyncio
    async def test_async_gen_empty(self) -> None:
        async def gen():
            return
            yield  # noqa: RUF027 — makes this an async generator, not a coroutine

        results = [x async for x in _to_async_gen(gen())]
        assert results == []

    @pytest.mark.asyncio
    async def test_async_gen_single_item(self) -> None:
        async def gen():
            yield "only"

        results = [x async for x in _to_async_gen(gen())]
        assert results == ["only"]
