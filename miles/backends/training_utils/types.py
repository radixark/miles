from dataclasses import dataclass
from enum import auto

from miles.utils.object_store import StoreObjectRef

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class TrainStepOutcome(StrEnum):
    NORMAL = auto()
    DISCARDED_SHOULD_RETRY = auto()


@dataclass(frozen=True)
class TrainStepOutput:
    outcome: TrainStepOutcome
    values: StoreObjectRef | None = None
