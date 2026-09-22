import random
from collections.abc import Callable
from dataclasses import dataclass

from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionEvidence, SoakActionRequest
from tests.utils.soak.core.views import SoakActionRecord
from tests.utils.soak.deploy.session import LauncherChain
from tests.utils.soak.deploy.types import DeploymentTarget
from tests.utils.soak.recipes.gsm8k import Gsm8kLaunchSpec

HOT_RESTART_FORM_NAME: str = "hot_restart"
TAKE_OVER_TIMEOUT_SECONDS: float = 1800.0
TAKE_OVER_POLL_INTERVAL_SECONDS: float = 10.0


def saved_iteration_after(action: SoakActionRecord) -> int:
    raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class HotRestartForm(BaseSoakActionForm):
    launch_spec: Gsm8kLaunchSpec
    event_log: EventLog
    max_allowed_rollout_id: int
    chain: LauncherChain

    @property
    def name(self) -> str:
        return HOT_RESTART_FORM_NAME

    @property
    def harms_target(self) -> bool:
        return False

    def maybe_create_request(
        self,
        *,
        target: DeploymentTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        raise NotImplementedError

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        raise NotImplementedError

    def is_recovered(self, *, action: SoakActionRecord, events: list[SoakEvent]) -> bool:
        raise NotImplementedError
