from __future__ import annotations

import logging
from abc import abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=BaseModel)


class StateMachineStepper(Generic[StateT]):
    """Pure-function state machine stepper: dispatch type(state) -> handler.

    Design contract:
    - A stepper holds immutable dependencies (protocol refs, config constants)
      and a handler registry. It must NOT accumulate evolving state across calls.
    - All evolving data lives in the State objects passed to __call__.
    - Per-call context slots (e.g. _tick_context, _call_trigger) that are
      overwritten before every invocation are permitted — they act as method
      parameters rather than long-lived state. Their values must not carry
      meaning from one call to the next.

    Subclasses implement _build_handlers() -> {type: async handler}.
    Terminal state = type(state) not in registry -> __call__ returns None.
    Subclasses may override __call__ to add pre-dispatch checks (e.g. timeout).
    """

    def __init__(self) -> None:
        self._handlers: dict[type, Callable[[StateT], Awaitable[StateT | None]]] = (
            self._build_handlers()
        )

    @abstractmethod
    def _build_handlers(self) -> dict[type, Callable[[StateT], Awaitable[StateT | None]]]: ...

    async def __call__(self, state: StateT) -> StateT | None:
        handler = self._handlers.get(type(state))
        if handler is None:
            return None
        result = await handler(state)
        if result is not None and type(result) is not type(state):
            logger.info(
                "%s %s -> %s",
                type(self).__name__,
                type(state).__name__,
                type(result).__name__,
            )
        return result


class StateMachine(Generic[StateT]):
    """The ONLY mutable state holder in the state machine architecture.

    This is the single place where state is stored and mutated. Everything else
    is either pure data (State) or a stateless function (Stepper).

    step() repeatedly calls the stepper until it returns None (no more transitions
    this tick). Each new state is recorded in state_history for debug/metrics.
    """

    _MAX_HISTORY = 128

    def __init__(self, *, initial_state: StateT, stepper: StateMachineStepper[StateT]) -> None:
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
    def stepper(self) -> StateMachineStepper[StateT]:
        return self._stepper

    async def step(self) -> None:
        """Run stepper until no more transitions this tick."""
        while True:
            new_state = await self._stepper(self._state)
            if new_state is None:
                break
            self._state_history.append(new_state)
            self._state = new_state
