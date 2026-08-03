import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.fsdp_utils import update_weight_utils
from miles.backends.training_utils.conn_status import ConnStatusManager


class _SessionEngine:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.calls = []
        self.submissions = []
        self.session_open = False

    def _record(self, name):
        self.submissions.append(name)
        self.calls.append(name)
        self.events.append(f"{self.name}.{name}")

    async def pause_generation(self):
        self._record("pause_generation")

    async def flush_cache(self):
        self._record("flush_cache")

    async def begin_weight_update(self, selector: str = "all"):
        self._record("begin_weight_update")
        assert not self.session_open
        self.session_open = True

    async def update_weights_from_tensor(self):
        self._record("update_weights_from_tensor")
        assert self.session_open, "update_weights_from_tensor requires an open begin_weight_update session"

    async def end_weight_update(self):
        self._record("end_weight_update")
        assert self.session_open
        self.session_open = False

    async def continue_generation(self):
        self._record("continue_generation")
        assert not self.session_open


class _FailingPhaseEngine(_SessionEngine):
    def __init__(self, name, events, failing_phase):
        super().__init__(name, events)
        self.failing_phase = failing_phase

    def _record(self, name):
        super()._record(name)
        if name == self.failing_phase:
            raise RuntimeError(f"{self.name} rejected {name}")


_SESSION_PHASES = [
    "pause_generation",
    "flush_cache",
    "begin_weight_update",
    "end_weight_update",
    "continue_generation",
]


class _SlowPhaseEngine(_SessionEngine):
    def __init__(self, name, events, slow_phase):
        super().__init__(name, events)
        self.slow_phase = slow_phase

    async def _delay_slow_phase(self, phase):
        if phase == self.slow_phase:
            await asyncio.sleep(0.05)

    async def pause_generation(self):
        await self._delay_slow_phase("pause_generation")
        await super().pause_generation()

    async def flush_cache(self):
        await self._delay_slow_phase("flush_cache")
        await super().flush_cache()

    async def begin_weight_update(self, selector: str = "all"):
        await self._delay_slow_phase("begin_weight_update")
        await super().begin_weight_update(selector)

    async def end_weight_update(self):
        await self._delay_slow_phase("end_weight_update")
        await super().end_weight_update()

    async def continue_generation(self):
        await self._delay_slow_phase("continue_generation")
        await super().continue_generation()


class _SessionAwareUpdater(update_weight_utils.UpdateWeight):
    def connect_rollout_engines(
        self,
        rollout_engines,
        engine_gpu_counts=None,
        engine_gpu_offsets=None,
    ):
        self.rollout_engines = rollout_engines

    def update_bucket_weights(self, named_tensors, weight_version=None):
        assert named_tensors
        self.last_named_tensors = named_tensors
        update_weight_utils.async_utils.wait_futures(
            [update_weight_utils.async_utils.submit(self.rollout_engines[0].update_weights_from_tensor())]
        )


def _make_updater(model, rollout_engines):
    updater = _SessionAwareUpdater(
        Namespace(update_weight_buffer_size=1024),
        SimpleNamespace(config=SimpleNamespace(model_type=""), state_dict=lambda: model),
    )
    updater.connect_rollout_engines(rollout_engines, None)
    return updater


def test_fsdp_weight_updates_run_inside_engine_session(monkeypatch):
    events = []
    engines = [_SessionEngine("engine0", events), _SessionEngine("engine1", events)]
    updater = _make_updater({"weight": torch.ones(1)}, engines)

    monkeypatch.setattr(update_weight_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_weight_utils.dist, "barrier", lambda **_kwargs: events.append("barrier"))
    monkeypatch.setattr(update_weight_utils, "get_gloo_group", lambda: object())
    monkeypatch.setattr(update_weight_utils, "gather_full_param", lambda param, async_op=False: param)

    updater.update_weights()

    assert events == [
        "engine0.pause_generation",
        "engine1.pause_generation",
        "engine0.flush_cache",
        "engine1.flush_cache",
        "engine0.begin_weight_update",
        "engine1.begin_weight_update",
        "barrier",
        "engine0.update_weights_from_tensor",
        "barrier",
        "engine0.end_weight_update",
        "engine1.end_weight_update",
        "engine0.continue_generation",
        "engine1.continue_generation",
        "barrier",
    ]
    assert engines[0].submissions == engines[0].calls
    assert engines[1].submissions == engines[1].calls


def test_fsdp_nonzero_rank_does_not_manage_engine_session(monkeypatch):
    events = []
    engine = _SessionEngine("engine0", events)
    updater = _make_updater({}, [engine])

    monkeypatch.setattr(update_weight_utils.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(update_weight_utils.dist, "barrier", lambda **_kwargs: events.append("barrier"))
    monkeypatch.setattr(update_weight_utils, "get_gloo_group", lambda: object())

    updater.update_weights()

    assert events == ["barrier", "barrier", "barrier"]
    assert engine.submissions == []


class TestUpdateWeight:
    @pytest.mark.parametrize("failing_phase", _SESSION_PHASES)
    def test_phase_failure_settles_all_engines_and_propagates(self, monkeypatch, failing_phase):
        """A failure in any session phase still lets the slower engine's request for that phase settle, then propagates."""
        events = []
        engines = [
            _FailingPhaseEngine("engine0", events, failing_phase),
            _SlowPhaseEngine("engine1", events, failing_phase),
        ]
        updater = _make_updater({"weight": torch.ones(1)}, engines)

        monkeypatch.setattr(update_weight_utils.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(update_weight_utils.dist, "barrier", lambda **_kwargs: events.append("barrier"))
        monkeypatch.setattr(update_weight_utils, "get_gloo_group", lambda: object())
        monkeypatch.setattr(update_weight_utils, "gather_full_param", lambda param, async_op=False: param)

        with pytest.raises(RuntimeError, match=f"engine0 rejected {failing_phase}"):
            updater.update_weights()

        reached_phases = _SESSION_PHASES[: _SESSION_PHASES.index(failing_phase) + 1]
        assert events[-2:] == [f"engine0.{failing_phase}", f"engine1.{failing_phase}"]
        assert [call for call in engines[0].calls if call in _SESSION_PHASES] == reached_phases
        assert engines[1].calls == reached_phases


def test_fsdp_weight_sync_casts_to_rollout_contract_dtypes(monkeypatch):
    events = []
    engine = _SessionEngine("engine0", events)
    fp32_value = torch.tensor([1.0 + 2**-20], dtype=torch.float32)
    updater = _make_updater(
        {
            "fp32_weight": fp32_value,
            "bf16_weight": fp32_value.clone(),
        },
        [engine],
    )
    updater.model._fsdp_sync_dtypes = {
        "fp32_weight": torch.float32,
        "bf16_weight": torch.bfloat16,
    }

    monkeypatch.setattr(update_weight_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_weight_utils.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(update_weight_utils, "get_gloo_group", lambda: object())
    monkeypatch.setattr(update_weight_utils, "gather_full_param", lambda param, async_op=False: param)

    updater.update_weights()

    synced = dict(updater.last_named_tensors)
    assert synced["fp32_weight"].dtype is torch.float32
    assert torch.equal(synced["fp32_weight"], fp32_value)
    assert synced["bf16_weight"].dtype is torch.bfloat16
    assert not torch.equal(synced["bf16_weight"].to(torch.float32), fp32_value)


class _RecordingWeightUpdater:
    def __init__(self) -> None:
        self.conn_status: ConnStatusManager = ConnStatusManager()
        self.connect_calls: list[list[object]] = []
        self.update_weights_calls: int = 0

    def connect_rollout_engines(
        self,
        rollout_engines: list[object],
        engine_gpu_counts: list[int] | None = None,
        engine_gpu_offsets: list[int] | None = None,
    ) -> None:
        self.connect_calls.append(list(rollout_engines))

    def update_weights(self) -> None:
        self.update_weights_calls += 1


def _make_updatable_engines(rollout_engines: list[object], *, has_new_engines: bool) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_engines=rollout_engines,
        has_new_engines=has_new_engines,
        engine_gpu_counts=[1] * len(rollout_engines),
        engine_gpu_offsets=list(range(len(rollout_engines))),
        snapshot_cell_id_to_hashes={},
    )


def test_fsdp_actor_connects_engines_once_across_consecutive_windows(monkeypatch):
    """Two weight-update windows over a stable engine set connect the rollout engines exactly once."""
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, ci_test=False)
    updater = _RecordingWeightUpdater()
    actor.weight_updater = updater
    engines: list[object] = [object(), object()]

    monkeypatch.setattr(actor_module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(actor_module, "clear_memory", lambda: None)

    actor.update_weights(_make_updatable_engines(engines, has_new_engines=True))
    actor.update_weights(_make_updatable_engines(engines, has_new_engines=False))

    assert updater.connect_calls == [engines]
    assert updater.update_weights_calls == 2
    assert not updater.conn_status.needs_reconnect({})


class _FakeFlattenedTensorBucket:
    def __init__(self, named_tensors):
        self.named_tensors = named_tensors

    def get_metadata(self):
        return [name for name, _ in self.named_tensors]

    def get_flattened_tensor(self):
        return torch.cat([tensor.flatten() for _, tensor in self.named_tensors])


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str=False):
        return "serialized-bucket"


class _TensorUpdateEngine:
    def __init__(self, response):
        self.response = response
        self.update_kwargs = []
        self.flush_cache_calls = 0

    async def update_weights_from_tensor(self, **kwargs):
        self.update_kwargs.append(kwargs)
        return self.response

    async def flush_cache(self):
        self.flush_cache_calls += 1


class TestUpdateWeightFromTensor:
    def test_raises_on_unsuccessful_engine_response(self, monkeypatch):
        """An engine answering success=false makes the bucket update raise its error message and skip the flush."""
        engine = _TensorUpdateEngine({"success": False, "error_message": "rollout engine rejected the bucket"})
        updater = update_weight_utils.UpdateWeightFromTensor(
            Namespace(update_weight_buffer_size=1024), torch.nn.Module()
        )
        updater._ipc_gather_src = 0
        updater._ipc_gather_group = object()
        updater._ipc_engine = engine

        def _fake_gather_object(obj, object_gather_list=None, dst=0, group=None):
            object_gather_list[0] = obj

        monkeypatch.setattr(update_weight_utils, "monkey_patch_torch_reductions", lambda: None)
        monkeypatch.setattr(update_weight_utils, "FlattenedTensorBucket", _FakeFlattenedTensorBucket)
        monkeypatch.setattr(update_weight_utils, "MultiprocessingSerializer", _FakeMultiprocessingSerializer)
        monkeypatch.setattr(update_weight_utils.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(update_weight_utils.dist, "get_world_size", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr(update_weight_utils.dist, "gather_object", _fake_gather_object)

        with pytest.raises(RuntimeError, match="rollout engine rejected the bucket"):
            updater.update_bucket_weights([("weight", torch.ones(2))], weight_version=7)

        assert [kwargs["weight_version"] for kwargs in engine.update_kwargs] == ["7"]
        assert engine.flush_cache_calls == 0


class _BroadcastHandle:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _DistributedUpdateEngine:
    def __init__(self, name, should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.requests = []

    async def update_weights_from_distributed(self, **kwargs):
        self.requests.append(kwargs)
        if self.should_fail:
            raise RuntimeError(f"{self.name} rejected the distributed update")


class TestUpdateWeightFromDistributed:
    def test_propagates_engine_request_failure(self, monkeypatch):
        """A failing update_weights_from_distributed request surfaces after every broadcast handle is waited on."""
        engines = [_DistributedUpdateEngine("engine0", should_fail=True), _DistributedUpdateEngine("engine1")]
        updater = object.__new__(update_weight_utils.UpdateWeightFromDistributed)
        updater.rollout_engines = engines
        updater._is_src_rank = True
        updater._group_name = "miles"
        updater._model_update_groups = object()

        handles = []

        def _fake_broadcast(tensor, src, group=None, async_op=False):
            handles.append(_BroadcastHandle())
            return handles[-1]

        monkeypatch.setattr(update_weight_utils.dist, "broadcast", _fake_broadcast)
        monkeypatch.setattr(update_weight_utils.dist, "get_world_size", lambda *_args, **_kwargs: 2)
        monkeypatch.setattr(update_weight_utils.torch.cuda, "empty_cache", lambda: None)

        with pytest.raises(RuntimeError, match="engine0 rejected the distributed update"):
            updater.update_bucket_weights([("a", torch.ones(2)), ("b", torch.zeros(3))], weight_version=3)

        assert len(handles) == 2
        assert all(handle.waited for handle in handles)
        assert [kwargs["names"] for kwargs in engines[1].requests] == [["a", "b"]]
