from __future__ import annotations

import logging
from abc import abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=BaseModel)
ContextT = TypeVar("ContextT")


class StateMachineStepper(Generic[StateT, ContextT]):
    """Pure-function state machine stepper: dispatch type(state) -> handler.

    Design contract:
    - A stepper holds immutable dependencies (protocol refs, config constants)
      and a handler registry. It must NOT accumulate evolving state across calls.
    - All evolving data lives in the State objects passed to __call__.
    - Per-call context is passed as an explicit parameter to __call__, never
      stored on the stepper instance.

    Subclasses implement _build_handlers() -> {type: async handler(state, context)}.
    Terminal state = type(state) not in registry -> __call__ returns None.
    Subclasses may override __call__ to add pre-dispatch checks (e.g. timeout).
    """

    def __init__(self) -> None:
        self._handlers: dict[type, Callable[[StateT, ContextT], Awaitable[StateT | None]]] = (
            self._build_handlers()
        )

    @abstractmethod
    def _build_handlers(self) -> dict[type, Callable[[StateT, ContextT], Awaitable[StateT | None]]]: ...

    async def __call__(self, state: StateT, context: ContextT) -> StateT | None:
        handler = self._handlers.get(type(state))
        if handler is None:
            return None
        result = await handler(state, context)
        if result is not None and result != state:
            logger.info(
                "%s %s -> %s",
                type(self).__name__,
                type(state).__name__,
                type(result).__name__,
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

    async def step(self, context: ContextT) -> None:
        """Run stepper until no more transitions this tick."""
        while True:
            new_state = await self._stepper(self._state, context)
            if new_state is None:
                break
            self._state_history.append(new_state)
            self._state = new_state
