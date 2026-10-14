from dataclasses import dataclass
from pathlib import Path

from tests.utils.soak.core.events import SoakObservationEvent
from tests.utils.soak.core.types import SoakObserver


@dataclass(frozen=True, kw_only=True)
class DeploymentObserver(SoakObserver):
    namespace: str
    release: str
    trainer_id: str
    checkpoint_dir: Path
    events_dir: Path

    async def observe(self) -> SoakObservationEvent:
        raise NotImplementedError
