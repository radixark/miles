"""Generic rollout handoff behavior, independent of any operation backend."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu")

import pytest
from tests.fast.ray.rollout.conftest import make_args

import miles.ray.rollout.rollout_manager as rollout_manager_module
from miles.rollout.base_types import RolloutFnHandoff, RolloutFnTrainOutput


class RecordingRolloutFn:
    def __init__(self, handoff: RolloutFnHandoff, abort_error: BaseException | None = None):
        self.handoff = handoff
        self.abort_error = abort_error
        self.aborts: list[tuple[RolloutFnHandoff, BaseException]] = []

    def __call__(self, _input):
        return RolloutFnTrainOutput(samples=[[object()]], handoff=self.handoff)

    async def abort_handoff(self, handoff, error):
        self.aborts.append((handoff, error))
        if self.abort_error is not None:
            raise self.abort_error


class BlockingAbortRolloutFn(RecordingRolloutFn):
    def __init__(self, handoff: RolloutFnHandoff):
        super().__init__(handoff)
        self.abort_started = asyncio.Event()
        self.finish_abort = asyncio.Event()

    async def abort_handoff(self, handoff, error):
        self.aborts.append((handoff, error))
        self.abort_started.set()
        await self.finish_abort.wait()


class FakeObjectStore:
    def __init__(self):
        self.error: BaseException | None = None
        self.values: list[dict] = []

    def put(self, value, value_spec):
        if self.error is not None:
            raise self.error
        self.values.append(value)
        return ("stored", value_spec)


def make_manager(monkeypatch, rollout_fn, *, delay_split=False):
    args = make_args(delay_split_train_data_by_dp=delay_split)
    manager = object.__new__(rollout_manager_module.RolloutManager.__ray_actor_class__)
    manager.args = args
    manager.servers = {}
    manager.rollout_id = -1
    manager.weight_version = None
    manager.train_parallel_config = {"dp_size": 1}
    manager.use_legacy_rollout_v1 = False
    manager.generate_rollout = rollout_fn
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.data_source = SimpleNamespace()
    manager._health_monitoring_resume = lambda: None

    store = FakeObjectStore()
    monkeypatch.setattr(rollout_manager_module, "timer", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(rollout_manager_module.dashboard_hooks, "register_engines", lambda *_a, **_k: None)
    monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", lambda *_a, **_k: None)
    monkeypatch.setattr(rollout_manager_module, "log_rollout_data", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rollout_manager_module,
        "postprocess_rollout_data",
        lambda *_a, **_k: (["flat-row"], {"derived": "metadata"}),
    )
    monkeypatch.setattr(
        rollout_manager_module,
        "convert_samples_to_train_data",
        lambda *_a, **_k: {
            "sample_indices": [7],
            "tokens": [[1]],
            # Deliberately conflicts with the control-plane receipt. The
            # manager must never reconstruct identity from converted tensors.
            "operation_by_lane": {0: "WRONG"},
        },
    )
    monkeypatch.setattr(
        rollout_manager_module,
        "split_train_data_by_dp",
        lambda _args, data, _config: ("split", data),
    )
    monkeypatch.setattr(rollout_manager_module.object_store, "get_instance", lambda: store)
    return manager, store


@pytest.mark.parametrize(
    ("stage", "delay_split"),
    [
        ("postprocess", False),
        ("timer", False),
        ("save", False),
        ("log", False),
        ("convert", False),
        ("split", False),
        ("store", True),
    ],
)
@pytest.mark.asyncio
async def test_every_downstream_failure_aborts_the_same_handoff(monkeypatch, stage, delay_split):
    error = OSError(f"{stage} failed")
    handoff = RolloutFnHandoff(receipt={"opaque_extension": {"receipt": "r1"}})
    rollout_fn = RecordingRolloutFn(handoff)
    manager, store = make_manager(monkeypatch, rollout_fn, delay_split=delay_split)

    def fail(*_args, **_kwargs):
        raise error

    if stage == "postprocess":
        monkeypatch.setattr(rollout_manager_module, "postprocess_rollout_data", fail)
    elif stage == "timer":

        class FailingTimer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                raise error

        monkeypatch.setattr(rollout_manager_module, "timer", lambda *_a, **_k: FailingTimer())
    elif stage == "save":
        monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", fail)
    elif stage == "log":
        monkeypatch.setattr(rollout_manager_module, "log_rollout_data", fail)
    elif stage == "convert":
        monkeypatch.setattr(rollout_manager_module, "convert_samples_to_train_data", fail)
    elif stage == "split":
        monkeypatch.setattr(rollout_manager_module, "split_train_data_by_dp", fail)
    else:
        store.error = error

    with pytest.raises(OSError) as caught:
        await manager.generate(rollout_id=1)

    assert caught.value is error
    assert rollout_fn.aborts == [(handoff, error)]


@pytest.mark.parametrize("delay_split", [False, True])
@pytest.mark.asyncio
async def test_success_forwards_opaque_metadata_without_interpretation(monkeypatch, delay_split):
    receipt = {
        "operation_ids": ["RIGHT"],
        "lease": {"dispatch_id": "lease-right", "bindings_by_operation": []},
    }
    handoff = RolloutFnHandoff(receipt=receipt)
    rollout_fn = RecordingRolloutFn(handoff)
    manager, _store = make_manager(monkeypatch, rollout_fn, delay_split=delay_split)

    pack = await manager.generate(rollout_id=2)

    assert pack["rollout_handoff"] is receipt
    assert rollout_fn.aborts == []
    assert pack["sample_indices"] == [7]
    assert pack["data_ref"][0] == ("stored" if delay_split else "split")


@pytest.mark.asyncio
async def test_abort_failure_does_not_mask_the_downstream_error(monkeypatch):
    handoff = RolloutFnHandoff(receipt={"opaque_extension": {"receipt": "r3"}})
    rollout_fn = RecordingRolloutFn(handoff, abort_error=RuntimeError("abort unavailable"))
    manager, _store = make_manager(monkeypatch, rollout_fn)
    error = OSError("conversion failed")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(rollout_manager_module, "convert_samples_to_train_data", fail)
    with pytest.raises(OSError) as caught:
        await manager.generate(rollout_id=3)

    assert caught.value is error
    assert rollout_fn.aborts == [(handoff, error)]


@pytest.mark.asyncio
async def test_aborter_cancellation_does_not_mask_the_downstream_error(monkeypatch):
    handoff = RolloutFnHandoff(receipt={"opaque_extension": {"receipt": "r4"}})
    rollout_fn = RecordingRolloutFn(handoff, abort_error=asyncio.CancelledError())
    manager, _store = make_manager(monkeypatch, rollout_fn)
    error = OSError("conversion failed")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(rollout_manager_module, "convert_samples_to_train_data", fail)
    with pytest.raises(OSError) as caught:
        await manager.generate(rollout_id=4)

    assert caught.value is error
    assert rollout_fn.aborts == [(handoff, error)]


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_cleanup_and_preserves_the_original_error(monkeypatch):
    handoff = RolloutFnHandoff(receipt={"opaque_extension": {"receipt": "r5"}})
    rollout_fn = BlockingAbortRolloutFn(handoff)
    manager, _store = make_manager(monkeypatch, rollout_fn)
    error = OSError("conversion failed")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(rollout_manager_module, "convert_samples_to_train_data", fail)
    task = asyncio.create_task(manager.generate(rollout_id=5))
    await rollout_fn.abort_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    rollout_fn.finish_abort.set()

    with pytest.raises(OSError) as caught:
        await task
    assert caught.value is error
    assert rollout_fn.aborts == [(handoff, error)]


@pytest.mark.asyncio
async def test_downstream_cancellation_aborts_then_propagates(monkeypatch):
    handoff = RolloutFnHandoff(receipt={"opaque_extension": {"receipt": "r6"}})
    rollout_fn = RecordingRolloutFn(handoff)
    manager, _store = make_manager(monkeypatch, rollout_fn)

    def cancel(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(rollout_manager_module, "convert_samples_to_train_data", cancel)
    with pytest.raises(asyncio.CancelledError):
        await manager.generate(rollout_id=6)

    assert len(rollout_fn.aborts) == 1
    assert rollout_fn.aborts[0][0] is handoff
    assert isinstance(rollout_fn.aborts[0][1], asyncio.CancelledError)


def test_rollout_manager_owns_no_tinker_dispatch_identity():
    import inspect

    import miles.ray.rollout.train_data_conversion as train_data_conversion

    source = inspect.getsource(rollout_manager_module)
    assert "tinker_dispatch" not in source
    assert "tinker_dispatch_summary" not in source
    assert not hasattr(train_data_conversion, "tinker_dispatch_summary")
