from __future__ import annotations

from miles.utils.ft.adapters.types import (
    STOP_TRAINING_TIMEOUT_SECONDS,
    JobStatus,
    MainJobProtocol,
    SubsystemActuatorProtocol,
)


class TrainingSubsystemActuator(SubsystemActuatorProtocol):
    """Training has no Level 1 restart capability yet.

    The sub-SM does not call stop()/start(); instead it transitions to
    RestartingMainJob state to request a job-level restart.
    """

    def __init__(self, main_job: MainJobProtocol) -> None:
        self._main_job = main_job

    async def start(self) -> str:
        raise NotImplementedError("Training subsystem-level restart not yet supported; use RestartingMainJob")

    async def stop(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None:
        raise NotImplementedError("Training subsystem-level restart not yet supported; use RestartingMainJob")

    async def get_status(self) -> JobStatus:
        return await self._main_job.get_status()
