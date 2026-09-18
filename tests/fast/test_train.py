import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tests.fast.fixtures.driver_fakes import FakeInferenceController, FakeRolloutExecutor, FakeTrainingModel


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def train_driver(monkeypatch):
    """Import the real driver with scoped stubs, without Ray/GPU initialization.

    Keep the real eval dispatcher and cadence helper so the existing scheduling
    regressions still exercise production code. Use --noconftest for a minimal
    environment: the repository-wide conftest imports optional GPU packages.
    """
    root = Path(__file__).resolve().parents[2]
    stubs = {
        "ray": {"kill": Mock()},
        "sglang.srt.constants": {
            "GPU_MEMORY_TYPE_CUDA_GRAPH": "cuda_graph",
            "GPU_MEMORY_TYPE_KV_CACHE": "kv_cache",
            "GPU_MEMORY_TYPE_WEIGHTS": "weights",
        },
        "miles.ray.placement_group": {
            "create_rollout_components": Mock(),
            "create_training_models": Mock(),
            "update_weights": Mock(),
        },
        "miles.ray.wiring": {"launch_worker_manager": Mock()},
        "miles.utils.object_store": {"init_instance": Mock()},
        "miles.utils.arguments": {"parse_args": Mock()},
        "miles.utils.audit_utils.process_identity": {"MainProcessIdentity": Mock()},
        "miles.utils.data": {"remove_rollout_data_refs": Mock()},
        "miles.utils.debug_utils.periodic_py_spy": {"maybe_start_periodic_pyspy_dump": Mock()},
        "miles.utils.ft_utils.api_server.server": {"start_api_server": Mock()},
        "miles.utils.ft_utils.mini_ft_controller": {"maybe_start_mini_ft_controller": Mock()},
        "miles.utils.logging_utils": {"configure_logger": Mock()},
        "miles.utils.tracking_utils.tracking": {"finish_tracking": Mock(), "init_tracking": Mock()},
        # Dependencies of the real misc.py; none is used by the cadence helper.
        "miles.utils.function_registry": {"load_function": Mock()},
        "miles.utils.http_utils": {"MILES_HOST_IP_ENV": "MILES_HOST_IP", "is_port_available": Mock()},
    }
    # Include parents to avoid side effects from package __init__ modules, and
    # restore every replaced module/attribute when the fixture exits.
    for name, attrs in stubs.items():
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent not in sys.modules:
                package = ModuleType(parent)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, parent, package)
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if len(parts) > 1:
            monkeypatch.setattr(sys.modules[name.rpartition(".")[0]], parts[-1], module, raising=False)
    for name in ("miles.ray.rollout.eval_dispatch", "miles.utils.misc", "miles.utils.lora"):
        module = _load_module(name, root / (name.replace(".", "/") + ".py"))
        monkeypatch.setitem(sys.modules, name, module)
    return _load_module("isolated_train_driver", root / "train.py")


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        colocate_memory_peak_device="cpu",
        debug_exit_after_rollout=None,
        eval_interval=None,
        eval_uses_snapshots=False,
        ft_components=[],
        fully_async=False,
        hf_checkpoint="/base/checkpoint",
        num_critic_only_steps=0,
        num_rollout=0,
        offload_rollout=False,
        offload_rollout_level="",
        offload_train=False,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=False,
        start_rollout_id=0,
        use_critic=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, events: list[str], train_driver: ModuleType
) -> SimpleNamespace:
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
        worker_manager=SimpleNamespace(
            dispose=SimpleNamespace(remote=AsyncMock(side_effect=lambda: events.append("worker_manager_dispose")))
        ),
    )

    async def create_rollout_components(_args: SimpleNamespace) -> tuple[Any, Any, int]:
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _controller: Any, _executor: Any) -> tuple[Any, Any]:
        return components.actor_model, components.critic_model

    async def update_weights(_model: Any, _executor: Any, rollout_id: int | None = None) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_driver, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_driver, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_driver, "launch_worker_manager", lambda _args: components.worker_manager)
    monkeypatch.setattr(
        train_driver.ray, "kill", Mock(side_effect=lambda *_args, **_kwargs: events.append("manager_kill"))
    )
    monkeypatch.setattr(train_driver.object_store, "init_instance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_driver, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    return components


async def test_continuous_rollout_runs_past_horizon_until_external_error(monkeypatch, train_driver):
    events = []
    args = _make_args(num_rollout=1, continuous_rollout=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    observed = []

    async def train_until_interrupted(rollout_id, *unused_args, **unused_kwargs):
        observed.append(rollout_id)
        if rollout_id == 3:
            raise RuntimeError("external stop")

    components.actor_model.train = train_until_interrupted
    with pytest.raises(RuntimeError, match="external stop"):
        await train_driver.train(args)
    assert observed == [0, 1, 2, 3]
    assert "update_weights:2" in events
    assert "worker_manager_dispose" in events


class TestEvalOnlyRun:
    async def test_eval_only_prepares_inference_and_runs_exactly_one_eval(
        self, monkeypatch: pytest.MonkeyPatch, train_driver
    ):
        """A run with no rollouts but an eval interval evaluates once and generates or trains nothing."""
        events: list[str] = []
        args = _make_args(num_rollout=0, eval_interval=2)
        components = _install_driver_fakes(monkeypatch, args, events, train_driver)

        await train_driver.train(args)

        assert events.count("prepare_eval") == 1
        assert events.count("eval:0") == 1
        assert events.index("prepare_eval") < events.index("eval:0")
        assert components.actor_model.trained == []
        assert not [event for event in events if event.startswith(("prepare_rollout", "generate_start"))]


class TestEvalBeforeTrain:
    async def test_fresh_run_evaluates_the_initial_checkpoint_at_step_zero(
        self, monkeypatch: pytest.MonkeyPatch, train_driver
    ):
        events: list[str] = []
        args = _make_args(num_rollout=2, eval_interval=1)
        _install_driver_fakes(monkeypatch, args, events, train_driver)

        await train_driver.train(args)

        assert events.index("eval:0") < events.index("prepare_rollout:0")

    async def test_resumed_run_evaluates_the_completed_rollout_not_the_next_one(
        self, monkeypatch: pytest.MonkeyPatch, train_driver
    ):
        """start_rollout_id is the loaded checkpoint's rollout plus one, so dispatching it
        would ask for a checkpoint that does not exist yet and, in staging mode, export
        into the directory the first resumed iteration goes on to overwrite."""
        events: list[str] = []
        args = _make_args(num_rollout=5, eval_interval=1, start_rollout_id=3)
        _install_driver_fakes(monkeypatch, args, events, train_driver)

        await train_driver.train(args)

        assert events.index("eval:2") < events.index("prepare_rollout:3")
        assert "eval:3" not in events[: events.index("prepare_rollout:3")]


class TestFinalEval:
    async def test_the_last_rollout_is_always_evaluated_even_off_cadence(
        self, monkeypatch: pytest.MonkeyPatch, train_driver
    ):
        """The final point carries force=True because training is over and backpressure is
        free, but the cadence check has to reach it first: with num_rollout not a multiple
        of eval_interval, the final weights would otherwise never be measured."""
        events: list[str] = []
        args = _make_args(num_rollout=3, eval_interval=2)
        _install_driver_fakes(monkeypatch, args, events, train_driver)

        await train_driver.train(args)

        assert [event for event in events if event.startswith("eval:")] == ["eval:0", "eval:1", "eval:2"]


class TestWeightEqualityCheck:
    async def test_weight_equality_check_is_routed_to_the_inference_controller(
        self, monkeypatch: pytest.MonkeyPatch, train_driver
    ):
        """--check-weight-update-equal must reach the inference controller with every comparison option intact."""
        events: list[str] = []
        args = _make_args(
            check_weight_update_equal=True,
            check_weight_update_allow_quant_error=True,
            check_weight_update_selector="layers.0",
            check_weight_update_skip_list=["lm_head", "embed_tokens"],
        )
        components = _install_driver_fakes(monkeypatch, args, events, train_driver)

        await train_driver.train(args)

        assert components.inference_controller.check_weights_calls == [
            dict(
                action="compare",
                allow_quant_error=True,
                selector="layers.0",
                skip_list=["lm_head", "embed_tokens"],
            )
        ]

    async def test_no_weight_comparison_without_the_flag(self, monkeypatch: pytest.MonkeyPatch, train_driver):
        """The comparison reloads weights on every engine, so an ordinary run must never trigger it."""
        events: list[str] = []
        args = _make_args(check_weight_update_equal=False)
        components = _install_driver_fakes(monkeypatch, args, events, train_driver)

        await train_driver.train(args)

        assert components.inference_controller.check_weights_calls == []


class TestTerminalLifecycle:
    async def test_train_disposes_all_created_component_controllers(
        self, monkeypatch: pytest.MonkeyPatch, train_driver
    ):
        """Every component the driver created must be disposed, or its watchers outlive the run."""
        events: list[str] = []
        args = _make_args(use_critic=True)
        _install_driver_fakes(monkeypatch, args, events, train_driver)

        await train_driver.train(args)

        assert _dispose_events(events) == _ALL_DISPOSE_EVENTS


@pytest.mark.parametrize("training_failed", [False, True])
async def test_eval_drain_precedes_cleanup_only_on_normal_exit(monkeypatch, train_driver, training_failed):
    events = []
    args = _make_args(num_rollout=1, use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    error = RuntimeError("reward filter failed")
    if training_failed:
        components.rollout_executor.generation_errors.append(error)
    drain = AsyncMock(side_effect=lambda: events.append("eval_drain"))
    monkeypatch.setattr(train_driver.EvalDispatcher, "drain", drain)

    if training_failed:
        with pytest.raises(RuntimeError) as caught:
            await train_driver.train(args)
        assert caught.value is error
        drain.assert_not_awaited()
    else:
        await train_driver.train(args)
        drain.assert_awaited_once()
        assert events.index("eval_drain") < events.index("executor_dispose")
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS


async def test_cleanup_failure_is_not_hidden_by_callers_exception_handler(monkeypatch, train_driver):
    args = _make_args()
    components = _install_driver_fakes(monkeypatch, args, [], train_driver)
    error = ValueError("cleanup failed")
    monkeypatch.setattr(components.actor_model, "dispose", AsyncMock(side_effect=error))
    try:
        raise RuntimeError("unrelated caller exception")
    except RuntimeError:
        with pytest.raises(ValueError) as caught:
            await train_driver.train(args)
    assert caught.value is error


async def test_remote_dispose_supports_non_coroutine_awaitables(monkeypatch, train_driver):
    events = []
    args = _make_args(use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)

    def submit_dispose():
        events.append("executor_dispose")
        result = asyncio.get_running_loop().create_future()
        result.set_result(None)
        return result

    monkeypatch.setattr(components.rollout_executor.dispose, "remote", submit_dispose)
    await train_driver.train(args)
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS


@pytest.mark.parametrize("stage", ["object_store", "tracking", "rollout_factory", "training_factory"])
async def test_early_failure_still_stops_new_worker_manager(monkeypatch, train_driver, stage):
    events = []
    args = _make_args(use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    error = RuntimeError(stage)
    if stage == "object_store":
        monkeypatch.setattr(train_driver.object_store, "init_instance", Mock(side_effect=error))
    elif stage == "tracking":
        monkeypatch.setattr(train_driver, "init_tracking", Mock(side_effect=error))
    else:
        factory = "create_rollout_components" if stage == "rollout_factory" else "create_training_models"
        monkeypatch.setattr(train_driver, factory, AsyncMock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        await train_driver.train(args)
    assert caught.value is error
    assert events[-2:] == ["worker_manager_dispose", "manager_kill"]
    train_driver.ray.kill.assert_called_once_with(components.worker_manager, no_restart=True)


async def test_failed_manager_launch_never_looks_up_or_kills_other_manager(monkeypatch, train_driver):
    events = []
    args = _make_args()
    _install_driver_fakes(monkeypatch, args, events, train_driver)
    error = RuntimeError("new manager failed to launch")
    monkeypatch.setattr(train_driver, "launch_worker_manager", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        await train_driver.train(args)
    assert caught.value is error
    assert not events
    train_driver.ray.kill.assert_not_called()


@pytest.mark.parametrize("training_failed", [False, True])
async def test_manager_dispose_failure_retains_handle_and_preserves_training_error(
    monkeypatch, train_driver, training_failed
):
    events = []
    args = _make_args(num_rollout=1, use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    original_error = RuntimeError("reward filter failed")
    if training_failed:
        components.rollout_executor.generation_errors.append(original_error)
    stop_error = ValueError("worker stop failed")

    async def fail_stop():
        events.append("worker_manager_dispose")
        raise stop_error

    monkeypatch.setattr(components.worker_manager.dispose, "remote", fail_stop)
    expected = original_error if training_failed else stop_error
    with pytest.raises(type(expected)) as caught:
        await train_driver.train(args)
    assert caught.value is expected
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS
    train_driver.ray.kill.assert_not_called()


async def test_manager_cleanup_completes_before_owner_is_killed(monkeypatch, train_driver):
    events = []
    args = _make_args()
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    started = asyncio.Event()
    release = asyncio.Event()

    async def stop_children():
        started.set()
        await release.wait()
        events.append("children_stopped")

    monkeypatch.setattr(components.worker_manager.dispose, "remote", stop_children)
    task = asyncio.create_task(train_driver.train(args))
    try:
        await asyncio.wait_for(started.wait(), 1)
        train_driver.ray.kill.assert_not_called()
        task.cancel()
        await asyncio.sleep(0)
        train_driver.ray.kill.assert_not_called()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events[-2:] == ["children_stopped", "manager_kill"]
    train_driver.ray.kill.assert_called_once_with(components.worker_manager, no_restart=True)


async def test_manager_timeout_does_not_kill_owner_before_children_stop(monkeypatch, train_driver, caplog):
    events = []
    args = _make_args()
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    monkeypatch.setattr(train_driver, "_DISPOSE_TIMEOUT_SECONDS", 0.01)
    finished = asyncio.Event()

    async def blocked_stop():
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    monkeypatch.setattr(components.worker_manager.dispose, "remote", blocked_stop)
    with pytest.raises(TimeoutError, match="worker_manager"):
        await train_driver.train(args)
    await asyncio.wait_for(finished.wait(), 1)
    train_driver.ray.kill.assert_not_called()
    assert "worker_manager" in caplog.text


_ALL_DISPOSE_EVENTS = [
    "executor_dispose",
    "inference_dispose",
    "actor_dispose",
    "critic_dispose",
    "worker_manager_dispose",
]


def _dispose_events(events):
    return [event for event in events if event.endswith("_dispose")]


@pytest.mark.parametrize("use_critic", [False, True])
async def test_reward_filter_error_disposes_components_without_masking_error(monkeypatch, train_driver, use_critic):
    events = []
    args = _make_args(num_rollout=1, use_critic=use_critic)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    error = RuntimeError("reward filter exhausted its sampling budget")
    components.rollout_executor.generation_errors.append(error)

    with pytest.raises(RuntimeError) as caught:
        await train_driver.train(args)

    assert caught.value is error
    assert _dispose_events(events) == [e for e in _ALL_DISPOSE_EVENTS if use_critic or e != "critic_dispose"]
    assert components.actor_model.trained == []


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("create_rollout_components", ["worker_manager_dispose"]),
        ("create_training_models", [*_ALL_DISPOSE_EVENTS[:2], "worker_manager_dispose"]),
        ("update_weights", _ALL_DISPOSE_EVENTS),
    ],
)
@pytest.mark.parametrize("cancelled", [False, True])
async def test_partial_initialization_cleans_only_returned_components(
    monkeypatch, train_driver, stage, expected, cancelled
):
    events = []
    args = _make_args(use_critic=True)
    _install_driver_fakes(monkeypatch, args, events, train_driver)
    error = asyncio.CancelledError(stage) if cancelled else RuntimeError(stage)
    monkeypatch.setattr(train_driver, stage, AsyncMock(side_effect=error))

    with pytest.raises(type(error)) as caught:
        await train_driver.train(args)

    assert caught.value is error
    assert _dispose_events(events) == expected


@pytest.mark.parametrize("stage", ["train", "drain"])
async def test_training_or_eval_drain_error_still_cleans_every_component(monkeypatch, train_driver, stage):
    events = []
    args = _make_args(num_rollout=1, use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    error = RuntimeError(stage)
    target = components.actor_model if stage == "train" else train_driver.EvalDispatcher
    monkeypatch.setattr(target, stage, AsyncMock(side_effect=error))

    with pytest.raises(RuntimeError) as caught:
        await train_driver.train(args)

    assert caught.value is error
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS


@pytest.mark.parametrize("component_name", ["rollout_executor", "inference_controller", "actor_model", "critic_model"])
@pytest.mark.parametrize("failure", ["async", "cancelled", "submission"])
@pytest.mark.parametrize("training_failed", [False, True])
async def test_disposal_failure_does_not_skip_later_components(
    monkeypatch, train_driver, component_name, failure, training_failed, caplog
):
    events = []
    args = _make_args(num_rollout=1, use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    original_error = RuntimeError("reward filter failed")
    if training_failed:
        components.rollout_executor.generation_errors.append(original_error)
    cleanup_error = (
        asyncio.CancelledError("dispose cancelled") if failure == "cancelled" else ValueError("dispose failed")
    )
    event_name = _ALL_DISPOSE_EVENTS[
        ["rollout_executor", "inference_controller", "actor_model", "critic_model"].index(component_name)
    ]

    async def fail_async():
        events.append(event_name)
        raise cleanup_error

    def fail_submission():
        events.append(event_name)
        raise cleanup_error

    component = getattr(components, component_name)
    dispose = fail_submission if failure == "submission" else fail_async
    if component_name == "rollout_executor":
        monkeypatch.setattr(component.dispose, "remote", dispose)
    else:
        monkeypatch.setattr(component, "dispose", dispose)
    expected_error = original_error if training_failed else cleanup_error

    with pytest.raises(type(expected_error)) as caught:
        await train_driver.train(args)

    # Python 3.10 may synthesize a new CancelledError at an asyncio.Task
    # boundary. Ordinary errors, especially the training failure, stay intact.
    if not isinstance(expected_error, asyncio.CancelledError):
        assert caught.value is expected_error
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS
    assert component_name in caplog.text


async def test_multiple_disposal_failures_propagate_first_after_all_attempts(monkeypatch, train_driver):
    events = []
    args = _make_args(use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    first_error = ValueError("executor failed")

    async def fail_executor():
        events.append("executor_dispose")
        raise first_error

    async def fail_inference():
        events.append("inference_dispose")
        raise RuntimeError("inference failed")

    monkeypatch.setattr(components.rollout_executor.dispose, "remote", fail_executor)
    monkeypatch.setattr(components.inference_controller, "dispose", fail_inference)
    with pytest.raises(ValueError) as caught:
        await train_driver.train(args)
    assert caught.value is first_error
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS


@pytest.mark.parametrize("training_failed", [False, True])
async def test_timeout_advances_even_when_disposer_delays_cancellation(
    monkeypatch, train_driver, training_failed, caplog
):
    events = []
    args = _make_args(num_rollout=1, use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    original_error = RuntimeError("reward filter failed")
    if training_failed:
        components.rollout_executor.generation_errors.append(original_error)
    release = asyncio.Event()
    cancel_seen = asyncio.Event()
    finished = asyncio.Event()
    monkeypatch.setattr(train_driver, "_DISPOSE_TIMEOUT_SECONDS", 0.01)

    async def slow_dispose():
        events.append("executor_dispose")
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancel_seen.set()
            await release.wait()
            raise ValueError("late cleanup error") from None
        finally:
            finished.set()

    monkeypatch.setattr(components.rollout_executor.dispose, "remote", slow_dispose)
    task = asyncio.create_task(train_driver.train(args))
    try:
        done, _ = await asyncio.wait([task], timeout=1)
        assert done, "cleanup waited indefinitely for cancellation acknowledgement"
        expected_type = RuntimeError if training_failed else TimeoutError
        with pytest.raises(expected_type) as caught:
            task.result()
        if training_failed:
            assert caught.value is original_error
        assert cancel_seen.is_set()
        assert _dispose_events(events) == _ALL_DISPOSE_EVENTS
        assert "timed out" in caplog.text
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.gather(task, return_exceptions=True)
    # Flush the completion callback that consumes the late exception.
    await asyncio.sleep(0)
    assert not train_driver._pending_dispose_tasks


async def test_task_cancellation_during_rollout_still_disposes_all_components(monkeypatch, train_driver):
    events = []
    args = _make_args(num_rollout=1, use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    started = asyncio.Event()

    async def blocked_get(_rollout_id):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(components.rollout_executor.get, "remote", blocked_get)
    task = asyncio.create_task(train_driver.train(args))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel("stop rollout")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS


@pytest.mark.parametrize("training_failed", [False, True])
async def test_repeated_cancellation_during_cleanup_is_deferred(monkeypatch, train_driver, training_failed):
    events = []
    args = _make_args(num_rollout=1, use_critic=True)
    components = _install_driver_fakes(monkeypatch, args, events, train_driver)
    original_error = RuntimeError("reward filter failed")
    if training_failed:
        components.rollout_executor.generation_errors.append(original_error)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_dispose():
        events.append("executor_dispose")
        started.set()
        await release.wait()

    monkeypatch.setattr(components.rollout_executor.dispose, "remote", blocked_dispose)
    task = asyncio.create_task(train_driver.train(args))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        for message in ("first cancel", "second cancel"):
            task.cancel(message)
            await asyncio.sleep(0)
            assert not task.done()
    finally:
        release.set()
    expected_type = RuntimeError if training_failed else asyncio.CancelledError
    with pytest.raises(expected_type) as caught:
        await task
    if training_failed:
        assert caught.value is original_error
    else:
        assert task.cancelled()
    assert _dispose_events(events) == _ALL_DISPOSE_EVENTS
