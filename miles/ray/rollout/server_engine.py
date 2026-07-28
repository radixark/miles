from __future__ import annotations

import logging

import ray

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.cell_state import (
    AddrInfo,
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedUninitialized,
    StateStopped,
)

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
        self._state = StateStopped()

    def mark_allocated_uninitialized(self, actor_handle: ray.actor.ActorHandle):
        self._change_state("mark_allocated", StateStopped, StateAllocatedUninitialized(actor_handle=actor_handle))

    def set_addressing(self, addr_info: AddrInfo) -> None:
        self._change_state(
            "set_addressing",
            StateAllocatedUninitialized,
            StateAllocatedUninitialized(actor_handle=self.actor_handle, addr_info=addr_info),
        )

    def mark_alive(self):
        self._change_state(
            "mark_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(actor_handle=self.actor_handle, addr_info=self.addr_info),
        )

    def mark_stopped(self):
        self._change_state("mark_stopped", (StateStopped, StateAllocatedBase), StateStopped())

    @property
    def actor_handle(self) -> ray.actor.ActorHandle:
        assert isinstance(self._state, StateAllocatedBase)
        return self._state.actor_handle

    @property
    def addr_info(self) -> AddrInfo:
        assert isinstance(self._state, StateAllocatedBase)
        assert self._state.addr_info is not None, f"{self._state=}"
        return self._state.addr_info

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.addr_info.server_url)

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        return isinstance(self._state, StateAllocatedAlive)

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        logger.info(f"{debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        logger.info(f"{debug_name} end new={self._state}")
