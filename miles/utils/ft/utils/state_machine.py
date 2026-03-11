from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=BaseModel)
ContextT = TypeVar("ContextT")
StateT_contra = TypeVar("StateT_contra", bound=BaseModel, contravariant=True)
ContextT_contra = TypeVar("ContextT_contra", contravariant=True)


class StateHandler(ABC, Generic[StateT_contra, ContextT_contra]):
    @abstractmethod
    async def step(self, state: StateT_contra, context: ContextT_contra) -> BaseModel | None: ...


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

    async def __call__(self, state: StateT, context: ContextT) -> StateT | None:
        if type(state) in self._terminal_states:
            return None

        if self._pre_dispatch is not None:
            result = await self._pre_dispatch(state, context)
            if result is not None:
                return result

        handler = self._handlers.get(type(state))
        if handler is None:
            raise TypeError(
                f"StateMachineStepper has no handler for state type "
                f"{type(state).__name__}; register it in handler_map"
            )
        result = await handler(state, context)
        if result is not None and result != state:
            logger.info(
                "%r -> %r",
                state,
                result,
            )
        return result


class StateMachine(Generic[StateT, ContextT]):
    """The ONLY mutable state holder in the state machine architecture.

    This is the single place where state is stored and mutated. Everything else
    is either pure data (State) or a stateless function (Stepper).

    step(context) repeatedly calls the stepper until it returns None (no more
    transitions this tick). Each new state is recorded in state_history for
    debug/metrics.
    """

    _MAX_HISTORY = 128

    def __init__(self, *, initial_state: StateT, stepper: StateMachineStepper[StateT, ContextT]) -> None:
        self._state = initial_state
        self._stepper = stepper
        self._state_history: deque[StateT] = deque(maxlen=self._MAX_HISTORY)

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def state_history(self) -> deque[StateT]:
        return self._state_history

    @property
    def stepper(self) -> StateMachineStepper[StateT, ContextT]:
        return self._stepper

    def force_state(self, new_state: StateT) -> None:
        """Externally inject a state (used by main SM to signal sub-SMs)."""
        self._state_history.append(new_state)
        self._state = new_state

    async def step(self, context: ContextT) -> None:
        """Run stepper until no more transitions this tick."""
        while True:
            new_state = await self._stepper(self._state, context)
            if new_state is None:
                break
            self._state_history.append(new_state)
            self._state = new_state
