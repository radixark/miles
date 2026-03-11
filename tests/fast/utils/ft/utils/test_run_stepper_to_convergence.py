"""Tests for run_stepper_to_convergence."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, ConfigDict

from miles.utils.ft.utils.state_machine import StateMachineStepper, run_stepper_to_convergence


# -- Dummy states --------------------------------------------------------------


class DummyState(BaseModel):
    model_config = ConfigDict(frozen=True)


class StateA(DummyState):
    pass


class StateB(DummyState):
    value: int = 0


class TerminalState(DummyState):
    pass


# -- Handlers ------------------------------------------------------------------


class StateAHandler:
    async def step(self, state: StateA, _ctx: None) -> DummyState:
        return StateB(value=1)


class StateBHandler:
    async def step(self, state: StateB, _ctx: None) -> DummyState | None:
        if state.value >= 3:
            return TerminalState()
        return StateB(value=state.value + 1)


class TerminalHandler:
    async def step(self, state: TerminalState, _ctx: None) -> None:
        return None


class NullHandler:
    """Always returns None — no transition."""

    async def step(self, state: StateA, _ctx: None) -> None:
        return None


class OscillatingAHandler:
    """A → B(0), creating an A-B-A-B... cycle."""

    async def step(self, state: StateA, _ctx: None) -> DummyState:
        return StateB(value=0)


class OscillatingBHandler:
    """B → A, creating an A-B-A-B... cycle."""

    async def step(self, state: StateB, _ctx: None) -> DummyState:
        return StateA()


class GenYieldsTwoHandler:
    """Async generator that yields two states in a single step."""

    async def step(self, state: StateA, _ctx: None):
        yield StateB(value=10)
        yield StateB(value=20)


# -- Stepper factories ---------------------------------------------------------


def _chain_stepper() -> StateMachineStepper[DummyState, None]:
    """A→B(1)→B(2)→B(3)→Terminal→(no transition)."""
    return StateMachineStepper(handler_map={
        StateA: StateAHandler,
        StateB: StateBHandler,
        TerminalState: TerminalHandler,
    })


def _null_stepper() -> StateMachineStepper[DummyState, None]:
    """StateA always returns None."""
    return StateMachineStepper(handler_map={StateA: NullHandler})


def _oscillating_stepper() -> StateMachineStepper[DummyState, None]:
    """A→B→A→B→... never converges."""
    return StateMachineStepper(handler_map={
        StateA: OscillatingAHandler,
        StateB: OscillatingBHandler,
    })


def _gen_stepper() -> StateMachineStepper[DummyState, None]:
    """StateA yields B(10), B(20) in one step; StateBHandler chains normally."""
    return StateMachineStepper(handler_map={
        StateA: GenYieldsTwoHandler,
        StateB: StateBHandler,
        TerminalState: TerminalHandler,
    })


def _terminal_stepper() -> StateMachineStepper[DummyState, None]:
    """StateA transitions, but TerminalState is configured as terminal."""
    return StateMachineStepper(
        handler_map={
            StateA: StateAHandler,
            StateB: StateBHandler,
            TerminalState: TerminalHandler,
        },
        terminal_states=frozenset({TerminalState}),
    )


# -- Tests ---------------------------------------------------------------------


class TestNoTransition:
    @pytest.mark.asyncio
    async def test_handler_returns_none_yields_nothing(self) -> None:
        results = [s async for s in run_stepper_to_convergence(_null_stepper(), StateA(), None)]
        assert results == []


class TestSingleStepTransition:
    @pytest.mark.asyncio
    async def test_single_state_change_yields_once(self) -> None:
        """A→B(1) in one dispatch, then B handler returns B(2), etc. Verify we get all intermediates."""
        stepper = _chain_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        assert len(results) == 4
        assert results[0] == StateB(value=1)
        assert results[1] == StateB(value=2)
        assert results[2] == StateB(value=3)
        assert results[3] == TerminalState()


class TestMultiDispatchConvergence:
    @pytest.mark.asyncio
    async def test_chain_a_to_terminal_yields_all_intermediates(self) -> None:
        """A→B(1)→B(2)→B(3)→Terminal: 4 stepper dispatches, 4 yields."""
        stepper = _chain_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        types = [type(s).__name__ for s in results]
        assert types == ["StateB", "StateB", "StateB", "TerminalState"]

    @pytest.mark.asyncio
    async def test_starting_from_mid_chain_converges(self) -> None:
        """B(2)→B(3)→Terminal: start from middle of chain."""
        stepper = _chain_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateB(value=2), None)]

        assert results == [StateB(value=3), TerminalState()]

    @pytest.mark.asyncio
    async def test_starting_at_terminal_yields_nothing(self) -> None:
        stepper = _terminal_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, TerminalState(), None)]
        assert results == []


class TestGeneratorHandler:
    @pytest.mark.asyncio
    async def test_gen_handler_yields_all_intermediate_states(self) -> None:
        """Gen handler yields B(10), B(20) in one step; then B(20)→B(21)→...→Terminal."""
        stepper = _gen_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        assert results[0] == StateB(value=10)
        assert results[1] == StateB(value=20)
        assert StateB(value=21) in results
        assert results[-1] == TerminalState()

    @pytest.mark.asyncio
    async def test_gen_handler_every_yield_is_visible(self) -> None:
        """Both yields from the gen handler appear — not just the last."""
        stepper = _gen_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        values = [s.value for s in results if isinstance(s, StateB)]
        assert 10 in values
        assert 20 in values


class TestMaxIterationsGuard:
    @pytest.mark.asyncio
    async def test_oscillating_stepper_stops_at_max_iterations(self) -> None:
        """A→B→A→B→... never converges; should stop at max_iterations."""
        stepper = _oscillating_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None, max_iterations=5)]

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_max_iterations_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = _oscillating_stepper()
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            _ = [s async for s in run_stepper_to_convergence(stepper, StateA(), None, max_iterations=3)]

        assert "hit max iterations (3)" in caplog.text

    @pytest.mark.asyncio
    async def test_normal_convergence_does_not_log_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = _chain_stepper()
        with caplog.at_level(logging.WARNING, logger="miles.utils.ft.utils.state_machine"):
            _ = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        assert "hit max iterations" not in caplog.text

    @pytest.mark.asyncio
    async def test_max_iterations_one_allows_single_dispatch(self) -> None:
        stepper = _chain_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None, max_iterations=1)]

        assert len(results) == 1
        assert results[0] == StateB(value=1)


class TestYieldOrdering:
    @pytest.mark.asyncio
    async def test_yields_are_in_transition_order(self) -> None:
        stepper = _chain_stepper()
        results = [s async for s in run_stepper_to_convergence(stepper, StateA(), None)]

        for i in range(len(results) - 1):
            if isinstance(results[i], StateB) and isinstance(results[i + 1], StateB):
                assert results[i].value < results[i + 1].value
