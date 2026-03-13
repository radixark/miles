from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")
StateT = TypeVar("StateT", bound=BaseModel)
ContextT = TypeVar("ContextT")
StateT_contra = TypeVar("StateT_contra", bound=BaseModel, contravariant=True)
ContextT_contra = TypeVar("ContextT_contra", contravariant=True)


async def _to_async_gen(raw: Awaitable[T] | AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    """Normalize a coroutine or async generator into a unified async generator."""
    if inspect.isasyncgen(raw):
        async for item in raw:
            yield item
    else:
        result = await raw
        yield result


class StateHandler(ABC, Generic[StateT_contra, ContextT_contra]):
    @abstractmethod
    async def step(
        self, state: StateT_contra, context: ContextT_contra
    ) -> BaseModel | AsyncGenerator[BaseModel, None] | None: ...


class StateMachineStepper(Generic[StateT, ContextT]):
    """Dispatch type(state) -> handler.step().

    Usage:
        stepper = StateMachineStepper(handler_map={
            DetectingAnomaly: DetectingAnomalyHandler,
            Recovering: RecoveringHandler,
        })

    Each handler class is a stateless namespace with a ``step(state, ctx)``
    method and optional private helpers. The class is instantiated once at
    stepper creation time.

    pre_dispatch: optional callback invoked before handler dispatch.
    If it returns a non-None state, that state is returned immediately
    (short-circuit). Use for cross-cutting concerns like timeout checks.
    """

    def __init__(
        self,
        handler_map: dict[type, type],
        *,
        terminal_states: frozenset[type] = frozenset(),
        pre_dispatch: Callable[[StateT, ContextT], Awaitable[StateT | None]] | None = None,
    ) -> None:
        self._handlers: dict[type, Callable[[StateT, ContextT], Awaitable[StateT | None]]] = {
            state_type: handler_cls().step for state_type, handler_cls in handler_map.items()
        }
        self._terminal_states = terminal_states
        self._pre_dispatch = pre_dispatch

    async def __call__(self, state: StateT, context: ContextT) -> AsyncGenerator[StateT, None]:
        if type(state) in self._terminal_states:
            return

        if self._pre_dispatch is not None:
            result = await self._pre_dispatch(state, context)
            if result is not None:
                yield result
                return

        handler = self._handlers.get(type(state))
        if handler is None:
            raise TypeError(
                f"StateMachineStepper has no handler for state type "
                f"{type(state).__name__}; register it in handler_map"
            )

        async for item in _to_async_gen(handler(state, context)):
            if item is not None:
                yield item


_MAX_CONVERGENCE_ITERATIONS = 50

ConvergenceFailureCallback = Callable[[object, int], None]


async def run_stepper_to_convergence(
    stepper: StateMachineStepper[StateT, ContextT],
    state: StateT,
    *,
    context_factory: Callable[[StateT], ContextT],
    max_iterations: int = _MAX_CONVERGENCE_ITERATIONS,
    on_convergence_failure: ConvergenceFailureCallback | None = None,
) -> AsyncGenerator[StateT, None]:
    current = state
    for _ in range(max_iterations):
        context = context_factory(current)
        had_transition = False
        async for new_state in stepper(current, context):
            if new_state != current:
                current = new_state
                had_transition = True
                yield current
        if not had_transition:
            return

    logger.warning(
        "run_stepper_to_convergence hit max iterations (%d), last state: %r",
        max_iterations,
        current,
    )
    if on_convergence_failure is not None:
        on_convergence_failure(current, max_iterations)


class StateMachine(Generic[StateT, ContextT]):
    """The ONLY mutable state holder in the state machine architecture.

    This is the single place where state is stored and mutated. Everything else
    is either pure data (State) or a stateless function (Stepper).

    step(context) repeatedly calls the stepper until it returns None (no more
    transitions this tick). Each new state is recorded in state_history for
    debug/metrics.
    """

    _MAX_HISTORY = 128

    # Accepted product decision: state is intentionally ephemeral (in-memory
    # only). If the controller process crashes, recovery progress is lost and
    # the next controller instance re-discovers faults from scratch. We do not
    # currently plan to add crash-recovery persistence here, so future audits
    # should not flag the lack of state checkpointing as a standalone issue
    # unless product requirements change.

    def __init__(
        self,
        *,
        initial_state: StateT,
        stepper: StateMachineStepper[StateT, ContextT],
        on_convergence_failure: ConvergenceFailureCallback | None = None,
    ) -> None:
        self._state = initial_state
        self._stepper = stepper
        self._state_history: deque[StateT] = deque(maxlen=self._MAX_HISTORY)
        self._on_convergence_failure = on_convergence_failure

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def state_history(self) -> deque[StateT]:
        return self._state_history

    async def step(self, context: ContextT) -> None:
        """Run stepper until no more transitions this tick."""
        for _ in range(_MAX_CONVERGENCE_ITERATIONS):
            had_transition = False
            async for new_state in self._stepper(self._state, context):
                if new_state != self._state:
                    logger.info("%r -> %r", self._state, new_state)
                    self._state_history.append(new_state)
                    self._state = new_state
                    had_transition = True
            if not had_transition:
                return

        logger.warning(
            "StateMachine.step hit max iterations (%d), last state: %r",
            _MAX_CONVERGENCE_ITERATIONS,
            self._state,
        )
        if self._on_convergence_failure is not None:
            self._on_convergence_failure(self._state, _MAX_CONVERGENCE_ITERATIONS)
