from __future__ import annotations

import logging

import ray
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


# NOTE: currently it is almost a dataclass without encapsulation to minimize code diff
#       (logic is batched currently while may be non-batched in the future)
#       ideally, it may encapsulate all actions and states, and ensure state transition
#       only happens after internal actions, while no external code can touch its internals
#       for example:
#         def __init__(...configs...)
#         def init(): _allocate_engine(); _mark_allocated(); _init_engine(); _mark_alive()
#         def stop(): _kill_engine(); _mark_stopped()
#       and external code cannot directly mutate the engines
#       this makes it more encapsulated, easier to reason about, and prevents state-resource inconsistency
class ServerEngine:
    def __init__(self):
        self._state = _StateStopped()

    def mark_allocated_uninitialized(self, actor_handle: ray.actor.ActorHandle):
        self._change_state("mark_allocated", _StateStopped, _StateAllocatedUninitialized(actor_handle=actor_handle))

    def mark_alive(self):
        self._change_state(
            "mark_alive", _StateAllocatedUninitialized, _StateAllocatedAlive(actor_handle=self.actor_handle)
        )

    def mark_stopped(self):
        self._change_state("mark_stopped", (_StateStopped, _StateAllocatedBase), _StateStopped())

    @property
    def actor_handle(self) -> ray.actor.ActorHandle:
        assert isinstance(self._state, _StateAllocatedBase)
        return self._state.actor_handle

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, _StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        return isinstance(self._state, _StateAllocatedAlive)

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[_State] | tuple[type[_State], ...],
        new_state: _State,
    ) -> None:
        logger.info(f"{debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        logger.info(f"{debug_name} end new={self._state}")


# ------------------------- states -----------------------------


class _StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class _StateStopped(_StateBase):
    pass


class _StateAllocatedBase(_StateBase):
    actor_handle: ray.actor.ActorHandle


class _StateAllocatedUninitialized(_StateAllocatedBase):
    pass


class _StateAllocatedAlive(_StateAllocatedBase):
    pass


_State = _StateStopped | _StateAllocatedUninitialized | _StateAllocatedAlive
