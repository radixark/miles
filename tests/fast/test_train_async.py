import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import train_async as train_async_driver
from tests.fast.fixtures.driver_fakes import FakeInferenceController, FakeRolloutExecutor, FakeTrainingModel


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

    async def create_rollout_components(_args: SimpleNamespace) -> tuple[Any, Any, int]:
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
