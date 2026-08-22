from types import SimpleNamespace
from typing import Any

import pytest

import train as train_driver
from tests.fast.fixtures.driver_fakes import (
    FakeInferenceController,
    FakeRemoteMethod,
    FakeRolloutExecutor,
    FakeTrainingModel,
)


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
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

    async def update_weights(
        _args: Any, _model: Any, _executor: Any, _inference_controller: Any, *, rollout_id: int | None = None
    ) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_driver, "init_orchestration_script", lambda _args: None)
    monkeypatch.setattr(train_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    return components


def _record_event_snapshots(rollout_executor: FakeRolloutExecutor, events: list[str]) -> None:
    async def snapshot_events(rollout_id: int) -> None:
        events.append(f"event_snapshot:{rollout_id}")

    rollout_executor.snapshot_events = FakeRemoteMethod(snapshot_events)


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


class TestCheckpointAuditSnapshot:
    async def test_a_saved_rollout_snapshots_post_update_audit_events(self, monkeypatch: pytest.MonkeyPatch):
        """A restorable checkpoint includes the checksum emitted by its successful weight publication."""
        events: list[str] = []
        args = _make_args(num_rollout=1, save_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)
        _record_event_snapshots(components.rollout_executor, events)

        await train_driver.train(args)

        assert events.index("actor_save:0") < events.index("update_weights:0") < events.index("event_snapshot:0")

    async def test_a_failed_weight_publication_keeps_only_the_early_checkpoint_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A failed publication cannot overwrite the checkpoint with audit events from an incomplete rollout."""
        events: list[str] = []
        args = _make_args(num_rollout=1, save_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)
        _record_event_snapshots(components.rollout_executor, events)

        async def fail_rollout_update(
            _args: Any,
            _model: Any,
            _executor: Any,
            _inference_controller: Any,
            *,
            rollout_id: int | None = None,
        ) -> None:
            if rollout_id == 0:
                raise RuntimeError("weight publication failed")

        monkeypatch.setattr(train_driver, "update_weights", fail_rollout_update)

        with pytest.raises(RuntimeError, match="weight publication failed"):
            await train_driver.train(args)

        assert "executor_save:0" in events
        assert "event_snapshot:0" not in events

    async def test_a_failed_post_update_snapshot_is_not_suppressed(self, monkeypatch: pytest.MonkeyPatch):
        """A failed audit refresh is a failed checkpointed rollout, not a silently incomplete success."""
        events: list[str] = []
        args = _make_args(num_rollout=2, save_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)

        async def fail_snapshot(rollout_id: int) -> None:
            events.append(f"event_snapshot:{rollout_id}")
            raise RuntimeError("event snapshot failed")

        components.rollout_executor.snapshot_events = FakeRemoteMethod(fail_snapshot)

        with pytest.raises(RuntimeError, match="event snapshot failed"):
            await train_driver.train(args)

        assert events.index("update_weights:0") < events.index("event_snapshot:0")
        assert "prepare_rollout:1" not in events
        assert not [event for event in events if event.endswith("_dispose")]

    async def test_a_rollout_without_a_checkpoint_does_not_snapshot_events(self, monkeypatch: pytest.MonkeyPatch):
        """A rollout that wrote no checkpoint must not invent a restorable audit snapshot directory."""
        events: list[str] = []
        args = _make_args(num_rollout=1, save_interval=None)
        components = _install_driver_fakes(monkeypatch, args, events)
        _record_event_snapshots(components.rollout_executor, events)

        await train_driver.train(args)

        assert "event_snapshot:0" not in events
