from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import NamedTuple

from miles.rollout.data_source import SourceReservation
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.utils.types import Sample


class FullyAsyncTerminalPendingError(RuntimeError):
    """Report that terminal state is not yet proven.

    The caller retains reservation ownership and may retry ``wait_terminal()``.
    """


class FullyAsyncExecutionSuccess(NamedTuple):
    """Report successful terminal execution for one exact attempt.

    Attributes:
        executor_receipt: The exact receipt passed to ``FullyAsyncExecutor.submit``.
        samples: Terminal generated output preserving reservation parent slots.
    """

    executor_receipt: ReservationExecutorReceipt
    samples: list[Sample | list[Sample]]


class FullyAsyncExecutionFailure(NamedTuple):
    """Report a terminal execution failure for one exact attempt.

    Attributes:
        executor_receipt: The exact receipt passed to ``FullyAsyncExecutor.submit``.
        error: The execution failure observed after all attempt work became terminal.
    """

    executor_receipt: ReservationExecutorReceipt
    error: BaseException


class FullyAsyncRetryReason(Enum):
    """Reason that terminal execution must replay its pristine reservation."""

    EXECUTION_ABORTED = auto()
    CANCELLATION_REQUESTED = auto()


class FullyAsyncExecutionRetry(NamedTuple):
    """Report terminal execution that produced no trainable group.

    Attributes:
        executor_receipt: The exact receipt passed to ``FullyAsyncExecutor.submit``.
        reason: Why the owner must replay the pristine source reservation.
    """

    executor_receipt: ReservationExecutorReceipt
    reason: FullyAsyncRetryReason


FullyAsyncExecutionOutcome = FullyAsyncExecutionSuccess | FullyAsyncExecutionFailure | FullyAsyncExecutionRetry


class FullyAsyncExecution(ABC):
    """Represent one exact submitted reservation attempt."""

    @abstractmethod
    def request_cancellation(self) -> None:
        """Request cancellation without claiming terminal completion.

        Implementations must tolerate repeated requests. Returning only records
        cancellation intent; the caller must still use ``wait_terminal()`` before
        releasing reservation ownership or capacity.
        """

    @abstractmethod
    async def wait_terminal(self) -> FullyAsyncExecutionOutcome:
        """Wait until all work owned by this attempt is terminal.

        Returns:
            A success, retry, or execution-failure outcome carrying the exact
            executor receipt supplied at submission.

        Raises:
            FullyAsyncTerminalPendingError: Terminal state is not yet proven.
                The caller retains ownership and may call this method again.
            Exception: Terminal observation itself violated its contract. Normal
                terminal execution failures are returned as
                ``FullyAsyncExecutionFailure``.
        """


class FullyAsyncExecutor(ABC):
    """Submit fully asynchronous attempts while preserving ownership boundaries."""

    @abstractmethod
    def submit(
        self,
        reservation: SourceReservation,
        executor_receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        """Submit one exact reservation attempt.

        Args:
            reservation: Pristine source reservation to execute without mutation.
            executor_receipt: Exact owner-issued receipt for this attempt.

        Returns:
            An execution whose terminal outcome carries ``executor_receipt`` by
            object identity.

        Raises:
            Exception: Submission failed before any work remained in flight. An
                implementation must not retain submitted work when it raises.
        """

    @abstractmethod
    async def close(self) -> None:
        """Stop accepting submissions and wait for every submitted attempt.

        Returns:
            None after all executor-owned work is terminal.
        """
