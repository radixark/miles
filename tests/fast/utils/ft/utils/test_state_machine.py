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
    async def test_self_transition_not_recorded_in_history(self) -> None:
        """Previously, yielding a state equal to the current state was still appended
        to _state_history, filling it with near-duplicate entries. Now self-transitions
        are skipped entirely (no history append, no state assignment, no had_transition).
        """
        stepper = _make_gen_stepper(SameStateGenHandler)
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        await machine.step(None)

        assert machine.state == StateA()
        assert len(machine.state_history) == 0

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
        results = [s async for s in run_stepper_to_convergence(_make_null_stepper(), StateA(), context_factory=lambda _: None)]
        assert results == []


class TestRunStepperToConvergenceChain:
    @pytest.mark.asyncio
    async def test_full_chain_yields_all_intermediates(self) -> None:
        """A→B(1)→B(2)→B(3)→Terminal: 4 stepper dispatches, 4 yields."""
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None)]

        assert len(results) == 4
        assert results[0] == StateB(value=1)
        assert results[1] == StateB(value=2)
        assert results[2] == StateB(value=3)
        assert results[3] == TerminalState()

    @pytest.mark.asyncio
    async def test_starting_mid_chain_converges(self) -> None:
        """B(2)→B(3)→Terminal: start from middle of chain."""
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateB(value=2), context_factory=lambda _: None)]

        assert results == [StateB(value=3), TerminalState()]

    @pytest.mark.asyncio
    async def test_starting_at_terminal_yields_nothing(self) -> None:
        stepper = _make_stepper(terminal_states=frozenset({TerminalState}))
        results = [s async for s in run_stepper_to_convergence(stepper, TerminalState(), context_factory=lambda _: None)]
        assert results == []

    @pytest.mark.asyncio
    async def test_yields_are_in_transition_order(self) -> None:
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None)]

        state_b_values = [s.value for s in results if isinstance(s, StateB)]
        assert state_b_values == sorted(state_b_values)


class TestRunStepperToConvergenceGenerator:
    @pytest.mark.asyncio
    async def test_gen_handler_yields_all_intermediate_states(self) -> None:
        """Gen handler yields B(1), B(2) in one step; then B handler chains to Terminal."""
        stepper = _make_gen_stepper(StateAGenHandler)
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None)]

        assert results[0] == StateB(value=1)
        assert results[1] == StateB(value=2)
        assert results[-1] == TerminalState()

    @pytest.mark.asyncio
    async def test_gen_handler_both_yields_visible(self) -> None:
        """Both yields from the gen handler appear — not just the last."""
        stepper = _make_gen_stepper(StateAGenHandler)
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None)]

        values = [s.value for s in results if isinstance(s, StateB)]
        assert 1 in values
        assert 2 in values


class TestRunStepperToConvergenceSameStateYield:
    @pytest.mark.asyncio
    async def test_same_state_yield_not_treated_as_transition(self) -> None:
        """run_stepper_to_convergence previously treated every yield as a
        transition. If a handler yielded the same state (e.g. no-op handler),
        the loop would never converge. Now it uses value equality like
        StateMachine.step, so same-state yields are skipped."""
        stepper = _make_gen_stepper(SameStateGenHandler)
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None)]

        assert results == []

    @pytest.mark.asyncio
    async def test_same_state_yield_does_not_hit_max_iterations(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Convergence should complete immediately without hitting the cap."""
        stepper = _make_gen_stepper(SameStateGenHandler)
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            _ = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None)]

        assert "hit max iterations" not in caplog.text


class TestRunStepperToConvergenceMaxIterations:
    @pytest.mark.asyncio
    async def test_oscillating_stops_at_max_iterations(self) -> None:
        """A→B→A→B→... should stop at max_iterations."""
        stepper = _make_oscillating_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None, max_iterations=5)]

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_max_iterations_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = _make_oscillating_stepper()
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            _ = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None, max_iterations=3)]

        assert "hit max iterations (3)" in caplog.text

    @pytest.mark.asyncio
    async def test_normal_convergence_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = _make_stepper()
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            _ = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None)]

        assert "hit max iterations" not in caplog.text

    @pytest.mark.asyncio
    async def test_max_iterations_one_yields_single_dispatch(self) -> None:
        """max_iterations=1 allows exactly one stepper dispatch."""
        stepper = _make_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None, max_iterations=1)]

        assert len(results) == 1
        assert results[0] == StateB(value=1)

    @pytest.mark.asyncio
    async def test_oscillating_yields_alternate_states(self) -> None:
        """Verify the oscillating pattern: A→B(0)→A→B(0)→A."""
        stepper = _make_oscillating_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), context_factory=lambda _: None, max_iterations=4)]

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


# -- Tests: convergence failure callback ---------------------------------------


class TestConvergenceFailureCallback:
    """Convergence failure previously only logged a warning, providing no
    mechanism for the controller to escalate or track repeated failures.
    Now both run_stepper_to_convergence and StateMachine.step call an
    optional on_convergence_failure callback when the iteration cap is hit."""

    @pytest.mark.asyncio
    async def test_run_stepper_calls_callback_on_max_iterations(self) -> None:
        stepper = _make_oscillating_stepper()
        callback_calls: list[tuple[object, int]] = []

        def on_failure(state: object, iterations: int) -> None:
            callback_calls.append((state, iterations))

        _ = [s async for s in run_stepper_to_convergence(
            stepper, StateA(),
            context_factory=lambda _: None,
            max_iterations=3,
            on_convergence_failure=on_failure,
        )]

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == 3

    @pytest.mark.asyncio
    async def test_run_stepper_no_callback_on_normal_convergence(self) -> None:
        stepper = _make_stepper()
        callback_calls: list[tuple[object, int]] = []

        def on_failure(state: object, iterations: int) -> None:
            callback_calls.append((state, iterations))

        _ = [s async for s in run_stepper_to_convergence(
            stepper, StateA(),
            context_factory=lambda _: None,
            on_convergence_failure=on_failure,
        )]

        assert callback_calls == []

    @pytest.mark.asyncio
    async def test_run_stepper_no_callback_when_none(self) -> None:
        """No error when on_convergence_failure is None (default)."""
        stepper = _make_oscillating_stepper()
        _ = [s async for s in run_stepper_to_convergence(
            stepper, StateA(),
            context_factory=lambda _: None,
            max_iterations=3,
            on_convergence_failure=None,
        )]

    @pytest.mark.asyncio
    async def test_state_machine_calls_callback_on_max_iterations(self) -> None:
        stepper = _make_oscillating_stepper()
        callback_calls: list[tuple[object, int]] = []

        def on_failure(state: object, iterations: int) -> None:
            callback_calls.append((state, iterations))

        machine = StateMachine(
            initial_state=StateA(),
            stepper=stepper,
            on_convergence_failure=on_failure,
        )
        await machine.step(None)

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == 50

    @pytest.mark.asyncio
    async def test_state_machine_no_callback_on_normal_convergence(self) -> None:
        stepper = _make_stepper()
        callback_calls: list[tuple[object, int]] = []

        def on_failure(state: object, iterations: int) -> None:
            callback_calls.append((state, iterations))

        machine = StateMachine(
            initial_state=StateA(),
            stepper=stepper,
            on_convergence_failure=on_failure,
        )
        await machine.step(None)

        assert callback_calls == []


# -- Tests: context_factory parameter ------------------------------------------


class _ContextAwareState(DummyState):
    """State that carries the context value it was created with, to verify context refresh."""
    ctx_value: int = 0


class _ContextAwareTerminal(DummyState):
    ctx_value: int = 0


class _ContextAwareHandler:
    """Handler that reads context value and embeds it in the next state.

    Transitions: ctx_value < 3 → _ContextAwareState(ctx_value=ctx+1)
                 ctx_value >= 3 → _ContextAwareTerminal(ctx_value=ctx)
    """

    async def step(self, state: _ContextAwareState, context: int) -> DummyState:
        if context >= 3:
            return _ContextAwareTerminal(ctx_value=context)
        return _ContextAwareState(ctx_value=context + 1)


class _ContextAwareTerminalHandler:
    async def step(self, state: _ContextAwareTerminal, context: int) -> None:
        return None


def _make_context_aware_stepper() -> StateMachineStepper[DummyState, int]:
    return StateMachineStepper(
        handler_map={
            _ContextAwareState: _ContextAwareHandler,
            _ContextAwareTerminal: _ContextAwareTerminalHandler,
        },
        terminal_states=frozenset({_ContextAwareTerminal}),
    )


class TestRunStepperToConvergenceContextFactory:
    """Previously run_stepper_to_convergence used the same context object
    for every iteration of the convergence loop. If a subsystem completed
    recovery and returned to its initial detecting state within one tick,
    the stale context (e.g. old job_status) would cause the same fault to
    be re-detected immediately, leading to duplicate recoveries and
    notifications. Now an optional context_factory callback refreshes the
    context before each iteration."""

    @pytest.mark.asyncio
    async def test_context_factory_called_each_iteration(self) -> None:
        """context_factory is invoked before each stepper dispatch, receiving
        the current state so it can build a fresh context."""
        factory_calls: list[DummyState] = []
        call_count = 0

        def context_factory(current_state: DummyState) -> int:
            nonlocal call_count
            factory_calls.append(current_state)
            call_count += 1
            return call_count

        stepper = _make_context_aware_stepper()
        results = [
            s async for s in run_stepper_to_convergence(
                stepper,
                _ContextAwareState(),
                context_factory=context_factory,
            )
        ]

        assert len(factory_calls) >= 2
        assert isinstance(factory_calls[0], _ContextAwareState)

    @pytest.mark.asyncio
    async def test_context_factory_provides_fresh_context_each_iteration(self) -> None:
        """Each iteration gets a fresh context value from the factory,
        not the stale initial context."""
        iteration_counter = 0

        def context_factory(current_state: DummyState) -> int:
            nonlocal iteration_counter
            iteration_counter += 1
            return iteration_counter

        stepper = _make_context_aware_stepper()
        results = [
            s async for s in run_stepper_to_convergence(
                stepper,
                _ContextAwareState(),
                context_factory=context_factory,
            )
        ]

        assert any(isinstance(s, _ContextAwareTerminal) for s in results)

    @pytest.mark.asyncio
    async def test_context_factory_prevents_stale_context_replay(self) -> None:
        """Core scenario: a handler completes recovery (state goes back to
        initial), and the refreshed context no longer triggers re-entry.

        Without context_factory, the stale context would cause the handler
        to re-trigger the same transition in the same tick."""
        transitions: list[tuple[str, int]] = []

        class _ReplayState(DummyState):
            pass

        class _RecoveredState(DummyState):
            pass

        class _ReplayHandler:
            async def step(self, state: _ReplayState, context: dict) -> DummyState | None:
                transitions.append(("replay", context["fault_count"]))
                if context["fault_count"] > 0:
                    return _RecoveredState()
                return None

        class _RecoveredHandler:
            async def step(self, state: _RecoveredState, context: dict) -> DummyState | None:
                transitions.append(("recovered", context["fault_count"]))
                return _ReplayState()

        stepper: StateMachineStepper = StateMachineStepper(handler_map={
            _ReplayState: _ReplayHandler,
            _RecoveredState: _RecoveredHandler,
        })

        call_count = 0

        def factory(current_state: DummyState) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"fault_count": 1}
            return {"fault_count": 0}

        results = [
            s async for s in run_stepper_to_convergence(
                stepper,
                _ReplayState(),
                context_factory=factory,
            )
        ]

        replay_triggers = [t for t in transitions if t[0] == "replay" and t[1] > 0]
        assert len(replay_triggers) == 1, (
            "Without context refresh, the stale fault_count=1 would cause "
            "multiple recovery triggers in the same tick"
        )
