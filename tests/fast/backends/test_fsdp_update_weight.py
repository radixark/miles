from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.fsdp_utils import update_weight_utils


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
