from dataclasses import dataclass
from enum import auto
from typing import TYPE_CHECKING

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


if TYPE_CHECKING:
    from miles.utils.ray_utils import Box


class TrainStepOutcome(StrEnum):
    NORMAL = auto()
    DISCARDED_SHOULD_RETRY = auto()


@dataclass(frozen=True)
class TrainStepOutput:
    outcome: TrainStepOutcome
    values: "Box | None" = None
