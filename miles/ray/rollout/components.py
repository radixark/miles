"""Role-separated construction of the rollout plane
(codex-rollout-fullparameter-design-0810 §4.3/§4.8).

Consumer-facing names are fixed NOW to the roles PR #1842 will ship —
``inference_controller`` (engine/router/weight-update ownership) and
``rollout_executor`` (rollout-fn execution/conversion) — while the current
concretes are ``Legacy...Adapter`` views over ONE combined RolloutManager
actor. When the split lands, only ``create_rollout_components`` changes:
construct the real InferenceController and RolloutExecutor (behind a thin
adapter if their invocation shape differs), and every call site keeps its
role variable. Deliberately not named ``InferenceController``/
``RolloutExecutor`` (the future classes must not collide) and not ``_tbd``
(Legacy states what the object actually is and when it dies).

The ports carry only what the tinker driver needs — no copy of the full
future public surface, and sampling/scoring never enters the executor."""

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
        """Per-rollout engine preparation/health handling (the PR #1842
        InferenceController responsibility). The driver calls this before
        every ``rollout_executor.generate(rollout_id)``; the legacy combined
        manager prepares inside ``generate()`` itself, so its adapter's
        implementation is a no-op."""
        ...


class RolloutExecutorPort(Protocol):
    async def generate(self, rollout_id: int): ...


class RolloutLifecyclePort(Protocol):
    async def dispose_once(self) -> None: ...


class LegacyInferenceControllerAdapter:
    """Inference-owner role view over the combined RolloutManager. The raw
    actor handle is private: the training-side weight-update wiring reaches
    it through ``RolloutComponents.weight_update_owner`` (an opaque factory
    product), never through this role object."""

    def __init__(self, manager) -> None:
        self._manager = manager

    async def get_inference_endpoint(self) -> InferenceEndpoint:
        host, port = await self._manager.get_router_address.remote()
        return InferenceEndpoint(host=host, port=port)

    async def prepare_rollout(self, rollout_id: int) -> None:
        """No-op today: the combined ``RolloutManager.generate()`` performs
        its own per-rollout preparation internally. The PR #1842 controller
        moves that preparation here, and the driver already calls it in the
        right place."""


class LegacyRolloutExecutorAdapter:
    """Execution role view over the same combined RolloutManager."""

    def __init__(self, manager) -> None:
        self._manager = manager

    async def generate(self, rollout_id: int):
        return await self._manager.generate.remote(rollout_id)


class LegacyRolloutLifecycle:
    """Exactly-once disposal of the SHARED underlying actor: two role views
    must never each dispose the same manager."""

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
    # Opaque owner/target the training actors wire their weight-update push
    # against (today: the combined RolloutManager actor handle). The driver
    # passes it to create_training_models verbatim and never introspects it;
    # PR #1842's factory hands out its real controller-owned target here.
    weight_update_owner: object

    async def dispose(self) -> None:
        await self.lifecycle.dispose_once()


def create_rollout_components(args, pg) -> RolloutComponents:
    """The one construction seam: today it builds one RolloutManager and two
    role views over it; after PR #1842 it builds the real controller/executor
    pair — call sites never change. The tinker driver has no epochs, so the
    manager's num_rollout_per_epoch is deliberately not carried."""
    from miles.ray.placement_group import create_rollout_manager

    rollout_manager, _num_rollout_per_epoch = create_rollout_manager(args, pg)
    return RolloutComponents(
        inference_controller=LegacyInferenceControllerAdapter(rollout_manager),
        rollout_executor=LegacyRolloutExecutorAdapter(rollout_manager),
        lifecycle=LegacyRolloutLifecycle(rollout_manager),
        weight_update_owner=rollout_manager,
    )
