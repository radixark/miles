import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionEvidence, SoakActionRequest, SoakTarget
from tests.utils.soak.core.views import SoakActionRecord, project_actions
from tests.utils.soak.ft.types import PoolResizedEvidence, PoolTarget, ResizeDetails

from miles.utils.test_utils.kubectl_reads import patch_replicas, read_replicas

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ResizePoolForm(BaseSoakActionForm):
    namespace: str
    workload: str
    cell_type: str
    sizes: tuple[int, ...] | None
    min_replicas: int
    max_replicas: int

    @property
    def name(self) -> str:
        name = f"resize:{self.cell_type}"
        if self.sizes is not None:
            name += ":sized"
        return name

    @property
    def harms_target(self) -> bool:
        return False

    def maybe_create_request(
        self,
        *,
        target: SoakTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        if not isinstance(target, PoolTarget) or target.identity != self.workload:
            return None

        if self.sizes is None:
            replicas = self._draw_replicas(target=target, rng=rng)
        else:
            replicas = self._next_size(target=target, events=events, sizes=self.sizes)
        if replicas is None:
            return None

        return SoakActionRequest(target=target, form_name=self.name, details=ResizeDetails(replicas=replicas))

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, ResizeDetails), f"Request {request.request_id} names no pool size"
        assert request.target.identity == self.workload
        replicas = request.details.replicas

        before = await asyncio.to_thread(read_replicas, namespace=self.namespace, workload=self.workload)
        await asyncio.to_thread(patch_replicas, namespace=self.namespace, workload=self.workload, replicas=replicas)
        after = await asyncio.to_thread(read_replicas, namespace=self.namespace, workload=self.workload)
        assert after == replicas, f"{self.workload} reads {after} replica(s) right after being resized to {replicas}"

        logger.info(f"Resized {self.workload} {before} -> {after}")
        report_applied(PoolResizedEvidence(replicas_before=before, replicas_after=after))

    def is_recovered(self, *, action: SoakActionRecord, events: list[SoakEvent]) -> bool:
        if action.applied is None:
            return False
        details = action.requested.request.details
        assert isinstance(details, ResizeDetails)

        return any(
            observation.timestamp > action.applied.timestamp
            and isinstance(target, PoolTarget)
            and target.identity == self.workload
            and target.ready
            and target.replicas == details.replicas
            and len(target.cell_ids) == details.replicas
            for observation in events
            if isinstance(observation, SoakObservationEvent)
            for target in observation.targets or []
        )

    def _draw_replicas(self, *, target: PoolTarget, rng: random.Random) -> int | None:
        candidates = [
            replicas
            for replicas in (target.replicas - 1, target.replicas + 1)
            if self.min_replicas <= replicas <= self.max_replicas
        ]
        return rng.choice(candidates) if candidates else None

    def _next_size(self, *, target: PoolTarget, events: list[SoakEvent], sizes: tuple[int, ...]) -> int | None:
        index = sum(
            1
            for action in project_actions(events).values()
            if action.requested.request.target.identity == target.identity
            and isinstance(action.requested.request.details, ResizeDetails)
        )
        return sizes[index] if index < len(sizes) else None
