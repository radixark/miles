from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.train.group import TrainerController
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.types import DeploymentIdentity
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

pytestmark = pytest.mark.asyncio


def _controller(identity: DeploymentIdentity) -> TrainerController:
    return TrainerController(
        deployment_identity=identity,
        cell_provider=MagicMock(spec=BaseWorkerProvider),
        cell_operations=MagicMock(spec=BaseCellOperations),
        trainer_id="actor",
        role="actor",
        with_ref=False,
    )


class TestDeploymentIdentity:
    async def test_the_identity_names_the_launch_this_controller_was_started_by(self):
        """A trainer answers for the deployment that launched it, not for the arguments a script hands it later."""
        identity = DeploymentIdentity(run_uuid="0123456789abcdef", deploy_component="trainer")

        assert await _controller(identity).get_deployment_identity() is identity
