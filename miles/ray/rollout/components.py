"""Role-separated rollout construction: PR #1842 role names now, Legacy adapters over one combined RolloutManager."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InferenceEndpoint:
    """Where sampling requests go (the SGLang router)."""

    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class InferenceControllerPort(Protocol):
    async def get_inference_endpoint(self) -> InferenceEndpoint: ...

    async def prepare_rollout(self, rollout_id: int) -> None:
        """Called before every generate; no-op in the legacy adapter (PR #1842 moves engine preparation here)."""
        ...


class RolloutExecutorPort(Protocol):
    async def generate(self, rollout_id: int): ...


class RolloutLifecyclePort(Protocol):
    async def dispose_once(self) -> None: ...


class LegacyInferenceControllerAdapter:
    """Inference-owner role view over the combined RolloutManager; the raw handle rides only weight_update_owner."""

    def __init__(self, manager) -> None:
        self._manager = manager

    async def get_inference_endpoint(self) -> InferenceEndpoint:
        host, port = await self._manager.get_router_address.remote()
        return InferenceEndpoint(host=host, port=port)

    async def prepare_rollout(self, rollout_id: int) -> None:
        """No-op: the combined manager prepares inside generate(); PR #1842 moves that preparation here."""


class LegacyRolloutExecutorAdapter:
    """Execution role view over the same combined RolloutManager."""

    def __init__(self, manager) -> None:
        self._manager = manager

    async def generate(self, rollout_id: int):
        return await self._manager.generate.remote(rollout_id)


class LegacyRolloutLifecycle:
    """Exactly-once disposal of the SHARED underlying actor: two role views must never each dispose it."""

    def __init__(self, manager) -> None:
        self._manager = manager
        self._disposed = False

    async def dispose_once(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        await self._manager.dispose.remote()


@dataclass
class RolloutComponents:
    inference_controller: InferenceControllerPort
    rollout_executor: RolloutExecutorPort
    lifecycle: RolloutLifecyclePort
    # Opaque weight-update owner/target (today the combined manager handle); passed verbatim, never introspected.
    weight_update_owner: object

    async def dispose(self) -> None:
        await self.lifecycle.dispose_once()


def create_rollout_components(args, pg) -> RolloutComponents:
    """One construction seam: legacy manager + role views today, PR #1842's pair later; call sites never change."""
    from miles.ray.placement_group import create_rollout_manager

    rollout_manager, _num_rollout_per_epoch = create_rollout_manager(args, pg)
    return RolloutComponents(
        inference_controller=LegacyInferenceControllerAdapter(rollout_manager),
        rollout_executor=LegacyRolloutExecutorAdapter(rollout_manager),
        lifecycle=LegacyRolloutLifecycle(rollout_manager),
        weight_update_owner=rollout_manager,
    )
