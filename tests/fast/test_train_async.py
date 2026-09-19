import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import train_async as train_async_driver
from tests.fast.fixtures.driver_fakes import (
    FakeInferenceController,
    FakeRemoteMethod,
    FakeRolloutExecutor,
    FakeTrainingModel,
)


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        colocate=False,
        debug_exit_after_rollout=None,
        eval_hf_dir=None,
        eval_interval=None,
        eval_max_in_flight=2,
        eval_overflow_policy="skip",
        eval_uses_snapshots=True,
        ft_components=[],
        hf_checkpoint=None,
        keep_old_actor=False,
        num_critic_only_steps=0,
        num_rollout=0,
        offload_train=False,
        overlap_model_initialization=False,
        save_hf=None,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=False,
        start_rollout_id=0,
        update_weights_interval=1,
        use_critic=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, events: list[str]
) -> SimpleNamespace:
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
        api_server_calls=[],
    )

    async def create_rollout_components(
        _args: SimpleNamespace, *, wait_for_inference_ready: bool = True
    ) -> tuple[Any, Any, int]:
        assert wait_for_inference_ready
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _controller: Any, _executor: Any) -> tuple[Any, Any]:
        return components.actor_model, components.critic_model

    async def update_weights(_model: Any, _executor: Any, rollout_id: int | None = None) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_async_driver, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_async_driver, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_async_driver, "launch_worker_manager", lambda _args: None)
    monkeypatch.setattr(train_async_driver.object_store, "init_instance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_async_driver, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_async_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_async_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_async_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_async_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_async_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_async_driver, "start_api_server", lambda **kwargs: components.api_server_calls.append(kwargs)
    )
    return components


def _install_initialization_gates(monkeypatch, args, events):
    components = _install_driver_fakes(monkeypatch, args, events)
    started = {role: asyncio.Event() for role in ["training", "inference"]}
    released = {role: asyncio.Event() for role in started}
    finished = {role: asyncio.Event() for role in started}
    cancelled = {role: asyncio.Event() for role in started}
    failures = {}

    async def load(role):
        started[role].set()
        try:
            await released[role].wait()
            if role in failures:
                raise failures[role]
            finished[role].set()
        except asyncio.CancelledError:
            cancelled[role].set()
            raise

    async def wait_for_ready():
        await load("inference")

    async def bind_eval_fleet(fleet):
        assert finished["inference"].is_set()
        assert fleet is components.inference_controller.eval_fleet
        events.append("eval_fleet_bound")

    components.inference_controller.eval_fleet = object()
    components.inference_controller.wait_for_ready = wait_for_ready
    components.rollout_executor.set_eval_fleet = FakeRemoteMethod(bind_eval_fleet)

    async def create_rollout_components(_args, *, wait_for_inference_ready=True):
        if wait_for_inference_ready:
            await wait_for_ready()
            await bind_eval_fleet(components.inference_controller.eval_fleet)
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args, _controller, _executor):
        await load("training")
        return components.actor_model, components.critic_model

    monkeypatch.setattr(train_async_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_async_driver, "create_training_models", create_training_models)
    return SimpleNamespace(
        started=started, released=released, finished=finished, cancelled=cancelled, failures=failures
    )


class TestModelInitialization:
    async def test_overlapping_loads_finish_before_weights_are_published(self, monkeypatch):
        """The first weight update must wait for both models and a bound evaluation fleet."""
        events = []
        args = _make_args(overlap_model_initialization=True)
        gates = _install_initialization_gates(monkeypatch, args, events)
        driver = asyncio.create_task(train_async_driver.train(args))
        await asyncio.wait_for(asyncio.gather(*(gate.wait() for gate in gates.started.values())), 3)
        gates.released["training"].set()
        await asyncio.wait_for(gates.finished["training"].wait(), 3)
        assert not driver.done()
        assert "eval_fleet_bound" not in events
        assert "update_weights:None" not in events
        gates.released["inference"].set()
        await asyncio.wait_for(driver, 3)
        assert events.index("eval_fleet_bound") < events.index("update_weights:None")

    async def test_default_loading_stays_serial(self, monkeypatch):
        """An unset optimization flag must preserve the original resource-loading order."""
        events = []
        args = _make_args()
        gates = _install_initialization_gates(monkeypatch, args, events)
        driver = asyncio.create_task(train_async_driver.train(args))
        await asyncio.wait_for(gates.started["inference"].wait(), 3)
        assert not gates.started["training"].is_set()
        gates.released["inference"].set()
        await asyncio.wait_for(gates.started["training"].wait(), 3)
        assert "eval_fleet_bound" in events
        assert "update_weights:None" not in events
        gates.released["training"].set()
        await asyncio.wait_for(driver, 3)

    @pytest.mark.parametrize("failed", ["training", "inference"], ids=["training-fails", "inference-fails"])
    async def test_failed_load_cancels_its_sibling_before_publication(self, monkeypatch, failed):
        """A failed initialization must propagate its error without leaving the sibling coroutine running."""
        events = []
        args = _make_args(overlap_model_initialization=True)
        gates = _install_initialization_gates(monkeypatch, args, events)
        gates.failures[failed] = ValueError(f"{failed} load failed")
        driver = asyncio.create_task(train_async_driver.train(args))
        await asyncio.wait_for(asyncio.gather(*(gate.wait() for gate in gates.started.values())), 3)
        gates.released[failed].set()
        with pytest.raises(ValueError, match=f"{failed} load failed"):
            await asyncio.wait_for(driver, 3)
        other = "inference" if failed == "training" else "training"
        assert gates.cancelled[other].is_set()
        assert "eval_fleet_bound" not in events
        assert "update_weights:None" not in events

    async def test_caller_cancellation_drains_both_loads(self, monkeypatch):
        """Cancelling startup must finish cancellation of both outstanding load coroutines."""
        events = []
        args = _make_args(overlap_model_initialization=True)
        gates = _install_initialization_gates(monkeypatch, args, events)
        driver = asyncio.create_task(train_async_driver.train(args))
        await asyncio.wait_for(asyncio.gather(*(gate.wait() for gate in gates.started.values())), 3)
        driver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driver
        assert all(gate.is_set() for gate in gates.cancelled.values())
        assert "update_weights:None" not in events


class TestApiServer:
    async def test_api_server_receives_the_refactored_driver_handles(self, monkeypatch: pytest.MonkeyPatch):
        """The API server acts on the live actor and inference controller, so it must be handed those objects."""
        events: list[str] = []
        args = _make_args(api_server_port=8123, ft_components=["rollout"])
        components = _install_driver_fakes(monkeypatch, args, events)

        await train_async_driver.train(args)

        (call,) = components.api_server_calls
        assert call["actor_model"] is components.actor_model
        assert call["inference_controller"] is components.inference_controller
        assert call["port"] == 8123
        assert call["ft_components"] == ["rollout"]

    async def test_no_api_server_without_a_port(self, monkeypatch: pytest.MonkeyPatch):
        """An unrequested API server would expose a control plane the operator never asked for."""
        events: list[str] = []
        args = _make_args(api_server_port=None)
        components = _install_driver_fakes(monkeypatch, args, events)

        await train_async_driver.train(args)

        assert components.api_server_calls == []


class TestWeightEqualityCheck:
    async def test_weight_equality_check_is_routed_to_the_inference_controller(self, monkeypatch: pytest.MonkeyPatch):
        """--check-weight-update-equal must reach the inference controller with every comparison option intact."""
        events: list[str] = []
        args = _make_args(
            check_weight_update_equal=True,
            check_weight_update_allow_quant_error=True,
            check_weight_update_selector="layers.0",
            check_weight_update_skip_list=["lm_head", "embed_tokens"],
        )
        components = _install_driver_fakes(monkeypatch, args, events)

        await train_async_driver.train(args)

        assert components.inference_controller.check_weights_calls == [
            dict(
                action="compare",
                allow_quant_error=True,
                selector="layers.0",
                skip_list=["lm_head", "embed_tokens"],
            )
        ]


class TestPipelinedGeneration:
    async def test_inflight_next_rollout_finishes_before_weight_publication(self, monkeypatch: pytest.MonkeyPatch):
        """Generation for the next rollout starts while this one trains, but must settle before new weights ship."""
        events: list[str] = []
        args = _make_args(num_rollout=2, update_weights_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)
        held_generation = asyncio.Event()
        components.rollout_executor.generation_gates[1] = held_generation

        driver = asyncio.create_task(train_async_driver.train(args))
        await asyncio.wait_for(components.actor_model.train_started[0].wait(), timeout=10)

        assert "generate_start:1" in events
        assert "generate_done:1" not in events
        assert "update_weights:0" not in events

        held_generation.set()
        await asyncio.wait_for(driver, timeout=10)

        assert events.index("generate_start:1") < events.index("actor_train:0")
        assert events.index("generate_done:1") < events.index("update_weights:0")
        assert components.actor_model.trained == [0, 1]


class TestTerminalLifecycle:
    async def test_async_train_drains_eval_and_disposes_all_component_controllers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A run ends only once its in-flight eval has settled and every component it created is disposed."""
        events: list[str] = []
        args = _make_args(use_critic=True, keep_old_actor=True, eval_interval=1, hf_checkpoint="/ckpt/hf")
        _install_driver_fakes(monkeypatch, args, events)

        await train_async_driver.train(args)

        assert "eval:0" in events
        assert sorted(event for event in events if event.endswith("_dispose")) == [
            "actor_dispose",
            "critic_dispose",
            "executor_dispose",
            "inference_dispose",
        ]
