from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import FakeInferenceTopologyProvider, make_args, make_rollout_config
from tests.fast.ray.rollout.nested_hook_fixtures import _RecordingSource

from miles.ray.rollout import rollout_executor as executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.rollout.session.types import SessionServerInstance
from miles.utils import object_store
from miles.utils.args.component_rollout import InferenceRuntimeImmutState, InferenceRuntimeMutState
from miles.utils.function_registry import function_registry
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
            inference_controller_provider=FakeInferenceTopologyProvider(
                InferenceRuntimeImmutState(engine_count=1, gpu_count=1)
            ),
        )

        async def resolve_router(args, *, router_providers) -> dict:
            return {}

        async def resolve_session(args, *, provider) -> None:
            args.session_server_instances = [SessionServerInstance(addr="eval-session:5000")]

        class StopAfterAddressResolution(Exception):
            pass

        def stop_tracking(*args, **kwargs) -> None:
            raise StopAfterAddressResolution

        monkeypatch.setattr(executor_module, "resolve_router_addrs", resolve_router)
        monkeypatch.setattr(executor_module, "wait_session_server_ready", resolve_session)
        monkeypatch.setattr(executor_module, "init_tracking", stop_tracking)

        with pytest.raises(StopAfterAddressResolution):
            await executor.init()

        assert args.session_server_instances == [SessionServerInstance(addr="eval-session:5000")]
        assert args.inference_runtime_mut_state == InferenceRuntimeMutState(engine_count=1, gpu_count=1)

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


def _fresh_executor() -> RolloutExecutor:
    return RolloutExecutor(
        args=make_args(),
        router_providers=[],
        session_server_provider=None,
        inference_controller_provider=FakeInferenceTopologyProvider(
            InferenceRuntimeImmutState(engine_count=1, gpu_count=1)
        ),
    )


async def test_disposing_an_executor_that_never_initialized_is_a_no_op() -> None:
    """The driver may dispose it after a startup failure elsewhere, before its init created anything to dispose."""
    executor = _fresh_executor()

    await executor.dispose()

    assert executor._init_once.state is InitState.NOT_INITED


async def test_disposing_an_executor_whose_init_failed_still_sweeps_the_output_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after init connected the output store must not leak what the run wrote to it."""
    executor = _fresh_executor()
    executor._delete_store_refs = MagicMock()
    swept = []
    monkeypatch.setattr(executor, "_sweep_output_store", lambda: swept.append(True))
    with pytest.raises(RuntimeError), executor._init_once.guarding():
        raise RuntimeError("data source failed to load")

    await executor.dispose()

    assert swept == [True]


class _CustomProducer(FullyAsyncRolloutFn):
    def __init__(self, input) -> None:
        pass


@pytest.mark.parametrize("fully_async", [False, True])
async def test_a_fully_async_subclass_is_loaded_only_under_fully_async(
    monkeypatch: pytest.MonkeyPatch, fully_async: bool
) -> None:
    """Without --fully-async the subclass would run as a plain plugin, past the mode's checks and driver scheduling."""
    monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "0")
    monkeypatch.setattr(object_store, "_INSTANCE", None)
    monkeypatch.setattr(executor_module, "load_rollout_function", lambda input, path: _CustomProducer(input))
    args = make_rollout_config(
        debug_train_only=True,
        enable_sample_ownership_checker=False,
        data_source_path="test-source",
        object_store_backend="ray",
        use_wandb=False,
        use_mlflow=False,
        fully_async=fully_async,
    )
    executor = RolloutExecutor(
        args=args, router_providers=[], session_server_provider=None, inference_controller_provider=object()
    )

    with function_registry.temporary("test-source", _RecordingSource):
        if fully_async:
            await executor.init()
            assert isinstance(executor.generate_rollout, _CustomProducer)
        else:
            with pytest.raises(ValueError, match="pass --fully-async"):
                await executor.init()
