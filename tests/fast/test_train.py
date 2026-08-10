from types import SimpleNamespace
from typing import Any

import pytest

import train as train_driver
from tests.fast.fixtures.driver_fakes import FakeInferenceController, FakeRolloutExecutor, FakeTrainingModel


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        debug_exit_after_rollout=None,
        eval_interval=None,
        ft_components=[],
        fully_async=False,
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
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, events: list[str]
) -> SimpleNamespace:
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
    )

    async def create_rollout_components(_args: SimpleNamespace) -> tuple[Any, Any, int]:
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _executor: Any) -> tuple[Any, Any]:
        return components.actor_model, components.critic_model

    async def update_weights(_model: Any, _executor: Any, rollout_id: int | None = None) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_driver, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_driver, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_driver, "launch_worker_manager", lambda _args: None)
    monkeypatch.setattr(train_driver.object_store, "init_instance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_driver, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    return components


class TestEvalOnlyRun:
    async def test_eval_only_prepares_inference_and_runs_exactly_one_eval(self, monkeypatch: pytest.MonkeyPatch):
        """A run with no rollouts but an eval interval evaluates once and generates or trains nothing."""
        events: list[str] = []
        args = _make_args(num_rollout=0, eval_interval=2)
        components = _install_driver_fakes(monkeypatch, args, events)

        await train_driver.train(args)

        assert events.count("prepare_eval") == 1
        assert events.count("eval:0") == 1
        assert events.index("prepare_eval") < events.index("eval:0")
        assert components.actor_model.trained == []
        assert not [event for event in events if event.startswith(("prepare_rollout", "generate_start"))]


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

        await train_driver.train(args)

        assert components.inference_controller.check_weights_calls == [
            dict(
                action="compare",
                allow_quant_error=True,
                selector="layers.0",
                skip_list=["lm_head", "embed_tokens"],
            )
        ]

    async def test_no_weight_comparison_without_the_flag(self, monkeypatch: pytest.MonkeyPatch):
        """The comparison reloads weights on every engine, so an ordinary run must never trigger it."""
        events: list[str] = []
        args = _make_args(check_weight_update_equal=False)
        components = _install_driver_fakes(monkeypatch, args, events)

        await train_driver.train(args)

        assert components.inference_controller.check_weights_calls == []


class TestTerminalLifecycle:
    async def test_train_disposes_all_created_component_controllers(self, monkeypatch: pytest.MonkeyPatch):
        """Every component the driver created must be disposed, or its watchers outlive the run."""
        events: list[str] = []
        args = _make_args(use_critic=True)
        _install_driver_fakes(monkeypatch, args, events)

        await train_driver.train(args)

        assert sorted(event for event in events if event.endswith("_dispose")) == [
            "actor_dispose",
            "critic_dispose",
            "executor_dispose",
            "inference_dispose",
        ]
