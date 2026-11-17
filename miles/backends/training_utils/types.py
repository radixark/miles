from dataclasses import dataclass

from miles.backends.training_utils.outcome import TrainStepOutcome
from miles.utils.object_store import StoreObjectRef

__all__ = ["TrainStepOutcome", "TrainStepOutput"]


@dataclass(frozen=True)
class TrainStepOutput:
    outcome: TrainStepOutcome
    values: StoreObjectRef | None = None
