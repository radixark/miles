from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    # TODO may refactor DataSource API
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


# subclassing for different data in the future
@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    # engine weight version, None before the first weight update
    weight_version: int | None = None

    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    generate_state: GenerateState | None = None
    weight_version: str | None = None
    hf_dir: str | None = None

    @property
    def evaluation(self):
        return True


class TrainBatchRollbackReason(Enum):
    """Reason that a manager could not settle a leased train batch."""

    HANDOFF_FAILED = auto()
    TRAINER_ADMISSION_FAILED = auto()


class TrainAdmissionHold(ABC):
    """Own one claim that keeps source and train-batch admission closed.

    A hold captures the execution frontier present when it is acquired. Waiting
    observes that frontier without consuming completed groups or settling a
    train-batch lease. Release is linear even when its owner reports a failure.
    """

    def __init__(self) -> None:
        self._release_attempted = False
        self._terminal_observed = False

    async def wait_terminal(self) -> None:
        """Wait until the execution frontier captured by this hold is terminal."""
        if self._release_attempted:
            raise RuntimeError("Train admission hold already has a release attempt.")
        await self._wait_terminal()
        self._terminal_observed = True

    @abstractmethod
    async def _wait_terminal(self) -> None:
        """Implement terminal observation for this hold's frontier."""

    def release(self) -> None:
        """Release this exact claim on source and train-batch admission."""
        if self._release_attempted:
            raise RuntimeError("Train admission hold already has a release attempt.")
        self._release_attempted = True
        self._release()

    @abstractmethod
    def _release(self) -> None:
        """Implement release after the handle is claimed."""


class WeightUpdateAdmissionHold(TrainAdmissionHold, ABC):
    """Admission hold that can fence a trainer-owned weight update.

    A plain :class:`TrainAdmissionHold` remains a compatible lifecycle contract
    for rollouts that only need to close source admission.  Implementations
    that participate in trainer weight-update sequencing opt into this explicit
    capability instead of inheriting an update API they cannot honor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._weight_update_recorded = False

    def record_weight_update(self, weight_version: int | None = None) -> None:
        """Record that a trainer-owned weight update completed under this hold.

        The marker is deliberately separate from release: a successful update
        advances the rollout's local admission generation before new source
        work may be admitted.  Callers must first observe the hold frontier so
        no pre-update execution remains in flight.
        """
        if self._release_attempted:
            raise RuntimeError("Train admission hold already has a release attempt.")
        if not self._terminal_observed:
            raise RuntimeError("Train admission hold must observe its terminal frontier before a weight update.")
        if self._weight_update_recorded:
            raise RuntimeError("Train admission hold already recorded a weight update.")
        self._record_weight_update(weight_version)
        self._weight_update_recorded = True

    @abstractmethod
    def _record_weight_update(self, weight_version: int | None = None) -> None:
        """Apply a completed trainer-owned weight update to rollout state.

        A lifecycle that has no update-aware state must opt out of the update
        admission API explicitly; silently reopening admission would be unsafe.
        """


class RolloutFnLifecycle(ABC):
    """Expose optional rollout ownership and resource lifecycle controls."""

    @abstractmethod
    async def prepare_checkpoint(self, rollout_id: int) -> None:
        """Require quiescent train-batch ownership before checkpoint publication."""

    @abstractmethod
    async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
        """Close admission and return its linear ownership claim."""

    @abstractmethod
    async def close(self) -> None:
        """Close resources after all train-batch leases settle."""


class TrainBatchLease(ABC):
    """Own a rollout batch until remote trainers acknowledge its publication.

    Args:
        rollout_id: Training rollout that requested the batch.

    A successful commit records that every required remote trainer acknowledged
    the exact published train-data result. Settlement may be attempted only
    once, including when its implementation raises, and either entry point
    consumes that single attempt: the blocking ``commit``/``rollback`` pair for
    callers that own no event loop, and the awaitable
    ``commit_async``/``rollback_async`` pair for callers that must keep their
    loop responsive while settlement runs elsewhere.
    """

    def __init__(self, rollout_id: int) -> None:
        self._rollout_id = rollout_id
        self._settlement_attempted = False

    @property
    def rollout_id(self) -> int:
        """Return the rollout that acquired this batch."""
        return self._rollout_id

    def commit(self) -> None:
        """Record successful publication of the complete train-data result.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        self._claim_settlement()
        self._commit()

    async def commit_async(self) -> None:
        """Record successful publication without blocking the calling event loop.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        self._claim_settlement()
        await self._commit_async()

    @abstractmethod
    def _commit(self) -> None:
        """Implement publication settlement after the lease is claimed."""

    async def _commit_async(self) -> None:
        """Await publication settlement; blocking implementations run inline."""
        self._commit()

    def rollback(self, reason: TrainBatchRollbackReason) -> None:
        """Return ownership after train-data publication fails.

        Args:
            reason: Why the manager could not publish the result.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        self._claim_settlement()
        self._rollback(reason)

    async def rollback_async(self, reason: TrainBatchRollbackReason) -> None:
        """Return ownership without blocking the calling event loop.

        Args:
            reason: Why the manager could not publish the result.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        self._claim_settlement()
        await self._rollback_async(reason)

    @abstractmethod
    def _rollback(self, reason: TrainBatchRollbackReason) -> None:
        """Implement publication recovery after settlement is claimed."""

    async def _rollback_async(self, reason: TrainBatchRollbackReason) -> None:
        """Await publication recovery; blocking implementations run inline."""
        self._rollback(reason)

    def _claim_settlement(self) -> None:
        if self._settlement_attempted:
            raise RuntimeError(f"Train batch lease for rollout {self.rollout_id} already has a settlement attempt.")
        self._settlement_attempted = True


# TODO make it frozen
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] | None = None


@dataclass
class LeasedRolloutFnTrainOutput(RolloutFnTrainOutput):
    """Carry ordinary train output data with its required settlement lease.

    Args:
        samples: Generated samples grouped by source prompt.
        metrics: Optional rollout metrics.
        lease: Ownership to settle after the manager publishes train data.
    """

    lease: TrainBatchLease = field(kw_only=True)


# TODO make it frozen
@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] | None = None


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


@dataclass(frozen=True)
class GenerateFnInput:
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    # One generate may lead to multiple samples, such as multi-agent, tree-like exploration, or
    # multi-turn with removing thinking tokens.
    samples: Sample | list[Sample]


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    """Legacy rollout function call interface. Used when MILES_USE_LEGACY_ROLLOUT_V1 is enabled."""
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
