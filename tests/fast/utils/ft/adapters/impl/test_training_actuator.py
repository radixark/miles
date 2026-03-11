from unittest.mock import AsyncMock

import pytest

from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator
from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol


@pytest.fixture
def mock_main_job() -> AsyncMock:
    return AsyncMock(spec=MainJobProtocol)


@pytest.fixture
def actuator(mock_main_job: AsyncMock) -> TrainingSubsystemActuator:
    return TrainingSubsystemActuator(main_job=mock_main_job)


class TestTrainingSubsystemActuator:
    @pytest.mark.asyncio
    async def test_stop_raises_not_implemented(self, actuator: TrainingSubsystemActuator) -> None:
        with pytest.raises(NotImplementedError, match="subsystem-level restart not yet supported"):
            await actuator.stop()

    @pytest.mark.asyncio
    async def test_start_raises_not_implemented(self, actuator: TrainingSubsystemActuator) -> None:
        with pytest.raises(NotImplementedError, match="subsystem-level restart not yet supported"):
            await actuator.start()

    @pytest.mark.asyncio
    async def test_get_status_delegates_to_main_job(
        self,
        actuator: TrainingSubsystemActuator,
        mock_main_job: AsyncMock,
    ) -> None:
        mock_main_job.get_job_status.return_value = JobStatus.RUNNING

        status = await actuator.get_status()

        assert status == JobStatus.RUNNING
        mock_main_job.get_job_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_status_propagates_all_statuses(
        self,
        actuator: TrainingSubsystemActuator,
        mock_main_job: AsyncMock,
    ) -> None:
        for expected_status in JobStatus:
            mock_main_job.get_job_status.return_value = expected_status

            status = await actuator.get_status()

            assert status == expected_status
