from types import SimpleNamespace
from typing import Any

import pytest
import ray
import train_multi_lora_async as multi_lora_driver
from tests.fast.fixtures.driver_fakes import (
    FakeInferenceController,
    FakeRemoteMethod,
    FakeRolloutExecutor,
    FakeTrainingModel,
)

from miles.utils.multi_lora import EmptyBatchTimeoutError

_ACTIVE_SNAPSHOT = {"pending": [], "active": ["alpha"], "retiring": [], "cleanup": []}
_EMPTY_SNAPSHOT = {"pending": [], "active": [], "retiring": [], "cleanup": []}


class FakeMultiLoRAController:
    def __init__(self, events: list[str], snapshots: list[dict[str, list[str]]]) -> None:
        self.events = events
        self.snapshots = snapshots
        self.registered_adapters: list[tuple[str, Any]] = []
        self.init = FakeRemoteMethod(self._start)
        self.stop = FakeRemoteMethod(self._stop)
        self.http_host = FakeRemoteMethod(self._http_host)
        self.api_port = FakeRemoteMethod(self._api_port)
        self.snapshot = FakeRemoteMethod(self._snapshot)
        self.register_adapter = FakeRemoteMethod(self._register_adapter)

    async def _start(self) -> None:
        self.events.append("controller_start")

    async def _stop(self) -> None:
        self.events.append("controller_stop")

    async def _http_host(self) -> str:
        return "127.0.0.1"

    async def _api_port(self) -> int:
        return 7000

    async def _snapshot(self) -> dict[str, list[str]]:
        self.events.append("snapshot")
        return self.snapshots.pop(0)

    async def _register_adapter(self, name: str, config: Any) -> None:
        self.registered_adapters.append((name, config))


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        colocate=False,
        multi_lora_adapters=[],
        multi_lora_idle_poll_s=0.01,
        multi_lora_service_mode=False,
        sglang_router_ip="10.0.0.9",
        sglang_router_port=9000,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    snapshots: list[dict[str, list[str]]],
) -> SimpleNamespace:
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        controller=FakeMultiLoRAController(events, snapshots),
    )

    async def create_rollout_components(_args: SimpleNamespace) -> tuple[Any, Any, int]:
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _executor: Any) -> tuple[Any, Any]:
        return components.actor_model, None

    async def update_weights(_model: Any, _executor: Any, rollout_id: int | None = None) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(multi_lora_driver, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(multi_lora_driver, "launch_worker_manager", lambda _args: None)
    monkeypatch.setattr(multi_lora_driver.object_store, "init_instance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(multi_lora_driver, "init_tracking", lambda _args: None)
    monkeypatch.setattr(multi_lora_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(multi_lora_driver, "get_multi_lora_controller", lambda: components.controller)
    monkeypatch.setattr(multi_lora_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(multi_lora_driver, "define_new_adapter_metrics", lambda _snapshot: None)
    monkeypatch.setattr(multi_lora_driver, "update_weights", update_weights)
    monkeypatch.setattr(multi_lora_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    return components


def _task_error(cause: Exception) -> ray.exceptions.RayTaskError:
    return ray.exceptions.RayTaskError(
        function_name="rollout_executor.get",
        traceback_str=f"{type(cause).__name__}: {cause}",
        cause=cause,
        proctitle="ray::RolloutExecutor.get",
        pid=1234,
        ip="127.0.0.1",
    )


class TestAdapterLifecycle:
    async def test_one_active_adapter_completes_through_the_refactored_components(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """One active adapter runs push, generate, train and save once, then the driver tears everything down."""
        events: list[str] = []
        args = _make_args()
        _install_driver_fakes(monkeypatch, events, snapshots=[_ACTIVE_SNAPSHOT, _ACTIVE_SNAPSHOT, _EMPTY_SNAPSHOT])

        await multi_lora_driver.main(args)

        assert events == [
            "controller_start",
            "snapshot",
            "actor_reconcile_adapters",
            "update_weights:None",
            "snapshot",
            "prepare_rollout:0",
            "generate_start:0",
            "generate_done:0",
            "actor_train:0",
            "actor_save:0",
            "snapshot",
            "executor_dispose",
            "inference_dispose",
            "actor_dispose",
            "controller_stop",
        ]


class TestEmptyBatchTimeout:
    async def test_empty_batch_timeout_retries_the_same_rollout_without_training(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A generation that timed out with no trainable groups is retried under the same rollout id."""
        events: list[str] = []
        args = _make_args()
        components = _install_driver_fakes(
            monkeypatch,
            events,
            snapshots=[_ACTIVE_SNAPSHOT, _ACTIVE_SNAPSHOT, _ACTIVE_SNAPSHOT, _ACTIVE_SNAPSHOT, _EMPTY_SNAPSHOT],
        )
        components.rollout_executor.generation_errors = [_task_error(EmptyBatchTimeoutError("no trainable groups"))]

        await multi_lora_driver.main(args)

        assert [event for event in events if event.startswith(("generate_", "actor_train", "actor_save"))] == [
            "generate_start:0",
            "generate_failed:0",
            "generate_start:0",
            "generate_done:0",
            "actor_train:0",
            "actor_save:0",
        ]
        assert components.actor_model.trained == [0]
        assert components.actor_model.saved == [0]

    async def test_an_unrelated_task_error_stops_the_run(self, monkeypatch: pytest.MonkeyPatch):
        """Retrying every failed generation task would hide a real rollout crash behind an endless loop."""
        events: list[str] = []
        args = _make_args()
        components = _install_driver_fakes(monkeypatch, events, snapshots=[_ACTIVE_SNAPSHOT, _ACTIVE_SNAPSHOT])
        components.rollout_executor.generation_errors = [_task_error(ValueError("rollout worker died"))]

        with pytest.raises(ray.exceptions.RayTaskError, match="rollout worker died"):
            await multi_lora_driver.main(args)

        assert components.actor_model.trained == []
