from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.init_once import InitOnce

pytestmark = pytest.mark.asyncio


def _inited_executor() -> RolloutExecutor:
    guard = InitOnce("RolloutExecutor")
    with guard.guarding():
        pass
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor._init_once = guard
    return executor


class TestInitRunsExactlyOnce:
    async def test_a_constructed_executor_reports_itself_uninitialized(self):
        """The constructor the run really uses is what has to leave the guard clear."""
        executor = RolloutExecutor(
            args=make_args(debug_train_only=True),
            router_providers=[],
            session_server_provider=None,
            inference_controller_provider=MagicMock(),
        )

        assert await executor.is_initialized() is False

    async def test_an_executor_that_ran_init_reports_itself_initialized(self):
        """The wait at the start of the rollout components only ends once this answer flips back."""
        assert await _inited_executor().is_initialized() is True

    async def test_a_second_init_is_refused(self):
        """An executor process the previous script initialized is about to be replaced, not re-initialized."""
        with pytest.raises(AssertionError, match="stale worker"):
            await _inited_executor().init()
