from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import rollout_executor as executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.init_once import InitOnce, InitState

pytestmark = pytest.mark.asyncio


def _inited_executor() -> RolloutExecutor:
    guard = InitOnce("RolloutExecutor")
    with guard.guarding():
        pass
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor._init_once = guard
    return executor


class TestInitRunsExactlyOnce:
    async def test_train_only_eval_resolves_addresses_in_the_executor_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The executor resolves session addresses on its own copy of train-only evaluation args."""
        args = make_args(debug_train_only=True, eval_num_gpus=1, use_session_server=True)
        executor = RolloutExecutor(
            args=args,
            router_providers=[],
            session_server_provider=MagicMock(),
            inference_controller_provider=MagicMock(),
        )

        async def resolve_router(args, *, router_providers) -> dict:
            return {}

        async def resolve_session(args, *, provider) -> None:
            args.session_server_addrs = ["eval-session:5000"]

        class StopAfterAddressResolution(Exception):
            pass

        def stop_tracking(*args, **kwargs) -> None:
            raise StopAfterAddressResolution

        monkeypatch.setattr(executor_module, "resolve_router_addrs", resolve_router)
        monkeypatch.setattr(executor_module, "wait_session_server_ready", resolve_session)
        monkeypatch.setattr(executor_module, "init_tracking", stop_tracking)

        with pytest.raises(StopAfterAddressResolution):
            await executor.init()

        assert args.session_server_addrs == ["eval-session:5000"]

    async def test_a_constructed_executor_reports_itself_uninitialized(self):
        """The constructor the run really uses is what has to leave the guard clear."""
        executor = RolloutExecutor(
            args=make_args(debug_train_only=True),
            router_providers=[],
            session_server_provider=None,
            inference_controller_provider=MagicMock(),
        )

        assert await executor.get_init_state() == InitState.NOT_INITED.value

    async def test_an_executor_that_ran_init_reports_itself_initialized(self):
        """The wait at the start of the rollout components only ends once this answer flips back."""
        assert await _inited_executor().get_init_state() == InitState.INITED.value

    async def test_a_second_init_is_refused(self):
        """An executor process the previous script initialized is about to be replaced, not re-initialized."""
        with pytest.raises(AssertionError, match="stale worker"):
            await _inited_executor().init()
