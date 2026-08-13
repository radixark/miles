from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils import update_weight_utils


class _RemoteMethod:
    def __init__(self, fn, submissions, name):
        self._fn = fn
        self._submissions = submissions
        self._name = name

    def remote(self, *args, **kwargs):
        self._submissions.append(self._name)
        return _RemoteRef(self._fn, args, kwargs, self._name)


class _RemoteRef:
    def __init__(self, fn, args, kwargs, name):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.name = name

    def resolve(self):
        return self._fn(*self._args, **self._kwargs)


class _SessionEngine:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.calls = []
        self.pause_modes = []
        self.submissions = []
        self.session_open = False
        self.pause_generation = self._remote(self._pause_generation, "pause_generation")
        self.flush_cache = self._remote(lambda: self._record("flush_cache"), "flush_cache")
        self.begin_weight_update = self._remote(self._begin_weight_update, "begin_weight_update")
        self.update_weights_from_tensor = self._remote(
            self._update_weights_from_tensor,
            "update_weights_from_tensor",
        )
        self.end_weight_update = self._remote(self._end_weight_update, "end_weight_update")
        self.continue_generation = self._remote(self._continue_generation, "continue_generation")

    def _remote(self, fn, name):
        return _RemoteMethod(fn, self.submissions, name)

    def _record(self, name):
        self.calls.append(name)
        self.events.append(f"{self.name}.{name}")

    def _pause_generation(self, mode):
        self.pause_modes.append(mode)
        self._record("pause_generation")

    def _begin_weight_update(self):
        self._record("begin_weight_update")
        assert not self.session_open
        self.session_open = True

    def _update_weights_from_tensor(self):
        self._record("update_weights_from_tensor")
        assert self.session_open, "update_weights_from_tensor requires an open begin_weight_update session"

    def _end_weight_update(self):
        self._record("end_weight_update")
        assert self.session_open
        self.session_open = False

    def _continue_generation(self):
        self._record("continue_generation")
        assert not self.session_open


class _SessionAwareUpdater(update_weight_utils.UpdateWeight):
    def connect_rollout_engines(
        self,
        rollout_engines,
        rollout_engine_lock,
        engine_gpu_counts=None,
        engine_gpu_offsets=None,
    ):
        self.rollout_engines = rollout_engines

    def update_bucket_weights(self, named_tensors, weight_version=None):
        assert named_tensors
        self.last_named_tensors = named_tensors
        update_weight_utils.ray.get(self.rollout_engines[0].update_weights_from_tensor.remote())


def _resolve_refs(value):
    if isinstance(value, list):
        return [ref.resolve() for ref in value]
    return value.resolve()


def _make_updater(model, rollout_engines, pause_generation_mode="retract"):
    updater = _SessionAwareUpdater(
        Namespace(
            pause_generation_mode=pause_generation_mode,
            update_weight_buffer_size=1024,
        ),
        SimpleNamespace(config=SimpleNamespace(model_type=""), state_dict=lambda: model),
    )
    updater.connect_rollout_engines(rollout_engines, None)
    return updater


@pytest.mark.parametrize("pause_generation_mode", ["abort", "retract", "in_place"])
def test_fsdp_weight_updates_run_inside_engine_session(monkeypatch, pause_generation_mode):
    events = []
    ray_get_batches = []
    engines = [_SessionEngine("engine0", events), _SessionEngine("engine1", events)]
    updater = _make_updater(
        {"weight": torch.ones(1)},
        engines,
        pause_generation_mode=pause_generation_mode,
    )

    def resolve_and_record_refs(value):
        refs = value if isinstance(value, list) else [value]
        ray_get_batches.append([ref.name for ref in refs])
        return _resolve_refs(value)

    monkeypatch.setattr(update_weight_utils.ray, "get", resolve_and_record_refs)
    monkeypatch.setattr(update_weight_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_weight_utils.dist, "barrier", lambda **_kwargs: events.append("barrier"))
    monkeypatch.setattr(update_weight_utils, "get_gloo_group", lambda: object())
    monkeypatch.setattr(update_weight_utils, "gather_full_param", lambda param, async_op=False: param)

    updater.update_weights()

    expected_flush_events = (
        []
        if pause_generation_mode == "in_place"
        else [
            "engine0.flush_cache",
            "engine1.flush_cache",
        ]
    )
    assert events == [
        "engine0.pause_generation",
        "engine1.pause_generation",
        *expected_flush_events,
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
    assert engines[0].pause_modes == [pause_generation_mode]
    assert engines[1].pause_modes == [pause_generation_mode]
    expected_ray_get_batches = [["pause_generation", "pause_generation"]]
    if pause_generation_mode != "in_place":
        expected_ray_get_batches.append(["flush_cache", "flush_cache"])
    expected_ray_get_batches.extend(
        [
            ["begin_weight_update", "begin_weight_update"],
            ["update_weights_from_tensor"],
            ["end_weight_update", "end_weight_update"],
            ["continue_generation", "continue_generation"],
        ]
    )
    assert ray_get_batches == expected_ray_get_batches
    assert engines[0].submissions == engines[0].calls
    assert engines[1].submissions == engines[1].calls


def test_fsdp_nonzero_rank_does_not_manage_engine_session(monkeypatch):
    events = []
    engine = _SessionEngine("engine0", events)
    updater = _make_updater({}, [engine])

    monkeypatch.setattr(update_weight_utils.ray, "get", _resolve_refs)
    monkeypatch.setattr(update_weight_utils.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(update_weight_utils.dist, "barrier", lambda **_kwargs: events.append("barrier"))
    monkeypatch.setattr(update_weight_utils, "get_gloo_group", lambda: object())

    updater.update_weights()

    assert events == ["barrier", "barrier", "barrier"]
    assert engine.submissions == []


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

    monkeypatch.setattr(update_weight_utils.ray, "get", _resolve_refs)
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
