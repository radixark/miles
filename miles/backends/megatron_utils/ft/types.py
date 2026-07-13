from enum import StrEnum, auto


class TrainStepOutcome(StrEnum):
    NORMAL = auto()
    DISCARDED_SHOULD_RETRY = auto()
