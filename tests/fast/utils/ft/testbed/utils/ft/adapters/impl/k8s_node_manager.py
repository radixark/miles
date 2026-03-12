from __future__ import annotations

import ray

from miles.utils.ft.adapters.types import NodeManagerProtocol
from tests.fast.utils.ft.utils.training_simulator import NodeManagerStateActor


class TestbedNodeManager(NodeManagerProtocol):
    """NodeManagerProtocol backed by a NodeManagerStateActor for cross-process access.

    Records mark_node_bad calls so that tests can assert which nodes
    were evicted. Serialized into FtControllerActor via cloudpickle.
    """

    def __init__(self, state_actor: ray.actor.ActorHandle) -> None:
        self._state = state_actor

    @classmethod
    def create(cls) -> TestbedNodeManager:
        state_actor = NodeManagerStateActor.remote()
        return cls(state_actor=state_actor)

    async def mark_node_bad(
        self,
        node_id: str,
        reason: str = "",
        node_metadata: dict[str, str] | None = None,
    ) -> None:
        await self._state.mark_bad.remote(node_id, reason, node_metadata)

    async def clear_bad_nodes(self) -> None:
        await self._state.clear_bad_nodes.remote()

    def was_ever_marked_bad(self, node_id: str) -> bool:
        return ray.get(self._state.was_ever_marked_bad.remote(node_id))

    @property
    def state_actor(self) -> ray.actor.ActorHandle:
        return self._state
