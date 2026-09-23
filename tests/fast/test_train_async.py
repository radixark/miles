import asyncio
import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import train_async as train_async_driver
from tests.fast.fixtures.driver_fakes import FakeInferenceController, FakeRolloutExecutor, FakeTrainingModel

from miles.ray.rollout import rollout_executor as executor_module
from miles.rollout.base_types import RolloutFnTrainOutput


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
        fully_async=False,
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

    async def create_rollout_components(_args: SimpleNamespace, *, checkpoint_replay=False) -> tuple[Any, Any, int]:
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _controller: Any, _executor: Any) -> tuple[Any, Any]:
        return components.actor_model, components.critic_model

    async def update_weights(_model: Any, _executor: Any, rollout_id: int | None = None) -> None:
        # A real publication spans awaits; yield so an un-awaited or overlapped update is observable.
        events.append(f"update_weights_start:{rollout_id}")
        await asyncio.sleep(0)
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
    monkeypatch.setattr(
        train_async_driver, "remove_rollout_data_refs", lambda _args, ref: events.append(f"consumed:{ref['data_ref']}")
    )
    monkeypatch.setattr(
        train_async_driver, "start_api_server", lambda **kwargs: components.api_server_calls.append(kwargs)
    )
    return components


def _consumed(events: list[str]) -> list[str]:
    return [event.removeprefix("consumed:") for event in events if event.startswith("consumed:")]


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
        assert events.index("generate_done:1") < events.index("update_weights_start:0")
        assert components.actor_model.trained == [0, 1]
        assert _consumed(events) == ["rollout-data-0", "rollout-data-1"]

    async def test_fully_async_next_drain_starts_after_weight_publication(self, monkeypatch: pytest.MonkeyPatch):
        """The persistent producer needs no lookahead drain that captures the previous weight version."""
        events: list[str] = []
        args = _make_args(fully_async=True, num_rollout=2, update_weights_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)

        await train_async_driver.train(args)

        assert events.index("actor_train:0") < events.index("update_weights:0")
        assert events.index("update_weights:0") < events.index("generate_start:1")
        assert "generate_start:2" not in events
        assert components.actor_model.trained == [0, 1]
        assert _consumed(events) == ["rollout-data-0", "rollout-data-1"]

    async def test_fully_async_keeps_lookahead_between_weight_updates(self, monkeypatch: pytest.MonkeyPatch):
        """A drain may still overlap training when that step cannot change the current weight version."""
        events: list[str] = []
        args = _make_args(fully_async=True, num_rollout=3, update_weights_interval=2)
        components = _install_driver_fakes(monkeypatch, args, events)

        await train_async_driver.train(args)

        assert events.index("generate_start:1") < events.index("actor_train:0")
        assert events.index("actor_train:1") < events.index("update_weights:1")
        assert events.index("update_weights:1") < events.index("generate_start:2")
        assert components.actor_model.trained == [0, 1, 2]
        assert _consumed(events) == ["rollout-data-0", "rollout-data-1", "rollout-data-2"]

    async def test_pipelined_publishes_on_interval_and_next_drain_follows_publication(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Weights ship only on interval steps; the lookahead after a publication must see the new weights."""
        events: list[str] = []
        args = _make_args(num_rollout=4, update_weights_interval=2)
        _install_driver_fakes(monkeypatch, args, events)

        await train_async_driver.train(args)

        assert [e for e in events if e.startswith("update_weights:")] == [
            "update_weights:None",
            "update_weights:1",
            "update_weights:3",
        ]
        assert events.index("generate_start:2") < events.index("actor_train:1")
        assert events.index("generate_done:2") < events.index("update_weights_start:1")
        assert events.index("update_weights:1") < events.index("generate_start:3")
        assert _consumed(events) == [f"rollout-data-{i}" for i in range(4)]


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


class TestPublicationAndReferences:
    async def test_fully_async_drain_waits_for_completed_publication(self, monkeypatch):
        args = _make_args(fully_async=True, num_rollout=2)
        events = []
        components = _install_driver_fakes(monkeypatch, args, events)
        publishing = asyncio.Event()
        published = asyncio.Event()
        captured_versions = []
        executor = SimpleNamespace(
            args=SimpleNamespace(load_debug_rollout_data=None),
            weight_version=0,
            use_legacy_rollout_v1=False,
            train_parallel_config={},
            _checkpoint_source=None,
        )

        def generate(input):
            captured_versions.append((input.rollout_id, input.weight_version))
            return RolloutFnTrainOutput(samples=[])

        executor.generate_rollout = generate
        monkeypatch.setattr(executor_module, "postprocess_rollout_data", lambda _args, data, **kw: (data, {}))
        monkeypatch.setattr(executor_module, "assert_samples_weight_version_sane", lambda *a, **kw: None)
        monkeypatch.setattr(executor_module.RolloutDataInjectionUtil, "should_inject", lambda *a: False)
        get = components.rollout_executor.get._fn

        async def update_weights(_model, _executor, rollout_id=None):
            if rollout_id == 0:
                publishing.set()
                await published.wait()
                executor.weight_version = 1

        async def drain(rollout_id):
            await executor_module.RolloutExecutor.__ray_actor_class__._get_rollout_data(executor, rollout_id)
            return await get(rollout_id)

        monkeypatch.setattr(train_async_driver, "update_weights", update_weights)
        monkeypatch.setattr(components.rollout_executor.get, "_fn", drain)
        driver = asyncio.create_task(train_async_driver.train(args))
        try:
            await asyncio.wait_for(publishing.wait(), timeout=10)
            # Allow an incorrectly detached update's caller to start the next drain.
            await asyncio.sleep(0)
            assert captured_versions == [(0, 0)]
            assert not driver.done()
            published.set()
            await asyncio.wait_for(driver, timeout=10)
            assert captured_versions == [(0, 0), (1, 1)]
        finally:
            published.set()
            driver.cancel()
            await asyncio.gather(driver, return_exceptions=True)

    @pytest.mark.parametrize("fully_async", [False, True])
    @pytest.mark.parametrize("update_weights_interval", [1, 2])
    async def test_consumed_batch_is_released_before_save(self, monkeypatch, fully_async, update_weights_interval):
        class Batch(dict):
            pass

        args = _make_args(
            fully_async=fully_async,
            num_rollout=3,
            save_interval=1,
            update_weights_interval=update_weights_interval,
        )
        events = []
        components = _install_driver_fakes(monkeypatch, args, events)
        batches = {}

        async def get(rollout_id):
            batch = Batch(data_ref=f"rollout-data-{rollout_id}")
            batches[rollout_id] = weakref.ref(batch)
            return batch

        async def train(rollout_id, batch):
            assert batch is batches[rollout_id]()
            assert batch["data_ref"] == f"rollout-data-{rollout_id}"

        async def save(rollout_id, force_sync=False):
            # asyncio can retain the completed Task's wakeup callback for one loop turn.
            await asyncio.sleep(0)
            gc.collect()
            assert batches[rollout_id]() is None
            assert f"consumed:rollout-data-{rollout_id}" in events

        monkeypatch.setattr(components.rollout_executor.get, "_fn", get)
        monkeypatch.setattr(components.actor_model, "train", train)
        monkeypatch.setattr(components.actor_model, "save_model", save)
        await train_async_driver.train(args)
