from __future__ import annotations

import ray
from tests.fast.utils.ft.utils.training_simulator import NotifierStateActor

from miles.utils.ft.adapters.types import NotifierProtocol


class TestbedNotifier(NotifierProtocol):
    """NotifierProtocol backed by a NotifierStateActor for cross-process access.

    Records send() calls so that tests can assert on notifications.
    Serialized into FtControllerActor via cloudpickle.
    """

    def __init__(self, state_actor: ray.actor.ActorHandle) -> None:
        self._state = state_actor

    @classmethod
    def create(cls) -> TestbedNotifier:
        state_actor = NotifierStateActor.remote()
        return cls(state_actor=state_actor)

    async def send(self, title: str, content: str, severity: str) -> None:
        await self._state.record.remote(title, content, severity)

    async def aclose(self) -> None:
        pass

    @property
    def calls(self) -> list[tuple[str, str, str]]:
        return ray.get(self._state.get_calls.remote())

    @property
    def state_actor(self) -> ray.actor.ActorHandle:
        return self._state
