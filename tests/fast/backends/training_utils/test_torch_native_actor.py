from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.training_utils.torch_native import actor as base_module
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.utils.memory_utils import move_optimizer_state

_MODULE = "miles.backends.training_utils.torch_native.actor"


@contextmanager
def _recording(stages, name):
    stages.append(f"enter:{name}")
    yield
    stages.append(f"exit:{name}")


@contextmanager
def _noop_timer(_name):
    yield


class _Profiler:
    def iterate_train_actor(self, it):
        return it

    def iterate_train_log_probs(self, it):
        return it

    step = MagicMock()


class _DataIterator:
    def __init__(self):
        self.resets = 0
        self.fetches = 0

    def reset(self):
        self.resets += 1
        return self


class _Provider(TorchNativeTrainRayActor):
    def __init__(self):
        self.args = SimpleNamespace(
            micro_batch_size=1,
            offload_train=False,
            debug_rollout_only=False,
            data_pad_size_multiplier=1,
            qkv_format="thd",
            ci_test=False,
        )
        self.model_parts = [MagicMock()]
        self.optimizers = [object()]
        self.hf_config = object()
        self.prof = _Profiler()
        self.runner = object()
        self.align_token_side_channel = lambda t, pad: t
        self._heartbeat = MagicMock()

    def step_runner(self):
        return self.runner


@pytest.fixture
def step(monkeypatch):
    stages: list = []
    calls = {"log_probs": [], "steps": [], "stages": stages}
    replay = SimpleNamespace(
        FALLTHROUGH="fallthrough",
        REPLAY_BACKWARD="replay_backward",
        stage=lambda name: _recording(stages, name),
        fill=MagicMock(side_effect=lambda *a, **k: stages.append("fill")),
        log_prob_stage=lambda args: "replay_forward",
        rewind=MagicMock(side_effect=lambda: stages.append("rewind")),
        reset=MagicMock(side_effect=lambda: stages.append("reset")),
    )
    calls["replay"] = replay
    monkeypatch.setattr(base_module, "routing_replay", replay)
    monkeypatch.setattr(base_module, "get_data_iterator", lambda args, parts, data: ([object()], [2]))
    monkeypatch.setattr(
        TorchNativeTrainRayActor,
        "_log_probs",
        lambda self, runner, it, n, store_prefix="": (
            calls["log_probs"].append((runner, store_prefix)) or {f"{store_prefix}log_probs": [1]}
        ),
    )
    monkeypatch.setattr(
        TorchNativeTrainRayActor,
        "_optimizer_steps",
        lambda self, runner, it, n, rollout_id: calls["steps"].append(runner),
    )
    monkeypatch.setattr(base_module, "compute_advantages_and_returns", lambda args, data: data.update(adv=True))
    monkeypatch.setattr(base_module, "log_rollout_data", lambda rid, args, data: None)
    monkeypatch.setattr(base_module, "timer", _noop_timer)
    return calls


def test_the_rollout_step_runs_ref_then_actor_then_optimizer_under_the_right_stages(step):
    actor = _Provider()
    actor.ref_runner = object()
    actor.ref_context = lambda: _recording(step["stages"], "ref")
    rollout_data = {}

    actor._train_core(rollout_id=3, rollout_data=rollout_data)

    assert step["stages"] == [
        "fill",
        "enter:fallthrough",
        "enter:ref",
        "exit:ref",
        "exit:fallthrough",
        "enter:replay_forward",
        "exit:replay_forward",
        "rewind",
        "enter:replay_backward",
        "exit:replay_backward",
        "reset",
    ]
    assert step["log_probs"] == [(actor.ref_runner, "ref_"), (actor.runner, "")]
    assert step["steps"] == [actor.runner]
    assert rollout_data == {"ref_log_probs": [1], "log_probs": [1], "adv": True}
    assert step["replay"].fill.call_args.kwargs["align"] is actor.align_token_side_channel


def test_move_optimizer_state_does_not_grow_state_for_parameters_that_have_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    model = torch.nn.Linear(4, 4, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    assert model.bias not in optimizer.state

    move_optimizer_state([optimizer], "cpu")

    assert model.bias not in optimizer.state
    assert optimizer.state[model.weight]["exp_avg"].device.type == "cpu"
