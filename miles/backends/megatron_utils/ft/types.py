from enum import auto

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class TrainStepOutcome(StrEnum):
    NORMAL = auto()
    DISCARDED_SHOULD_RETRY = auto()
