"""The shared RL step of the torch-native backends, driven through a fake provider."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils import torch_native_actor as base_module
from miles.backends.training_utils.torch_native_actor import TorchNativeTrainRayActor

_MODULE = "miles.backends.training_utils.torch_native_actor"


def _fake_replay(stages: list):
    @contextmanager
    def stage(name):
        stages.append(f"enter:{name}")
        yield
        stages.append(f"exit:{name}")

    return SimpleNamespace(
        FALLTHROUGH="fallthrough",
        RECORD="record",
        REPLAY_FORWARD="replay_forward",
        REPLAY_BACKWARD="replay_backward",
        stage=stage,
        fill=MagicMock(side_effect=lambda *a, **k: stages.append("fill")),
        log_prob_stage=lambda args: "replay_forward",
        rewind=MagicMock(side_effect=lambda: stages.append("rewind")),
        reset=MagicMock(side_effect=lambda: stages.append("reset")),
    )


class _Provider(TorchNativeTrainRayActor):
    def __init__(self, stages, *, with_ref):
        self.args = SimpleNamespace(micro_batch_size=1, offload_train=False, debug_rollout_only=False)
        self.model_parts = [object()]
        self.optimizers = [object()]
        self.prof = MagicMock()
        self.routing_replay = _fake_replay(stages)
        self.runner = object()
        self.align_token_side_channel = lambda t, pad: t
        self.after_rollout_calls = []
        self._heartbeat = MagicMock()
        if with_ref:
            self.ref_runner = object()
            self.ref_context = lambda: _recording_context(stages, "ref")

    def step_runner(self):
        return self.runner

    def after_rollout(self, rollout_id, rollout_data):
        self.after_rollout_calls.append(rollout_id)


@contextmanager
def _recording_context(stages, name):
    stages.append(f"enter:{name}")
    yield
    stages.append(f"exit:{name}")


@contextmanager
def _noop_timer(_name):
    yield


@pytest.fixture
def loop(monkeypatch):
    calls = {"log_probs": [], "steps": []}
    iterator = object()
    monkeypatch.setattr(base_module, "get_data_iterator", lambda args, parts, data: ([iterator], [2]))
    monkeypatch.setattr(
        base_module,
        "run_log_probs",
        lambda args, it, n, runner, *, profiler, store_prefix="": (
            calls["log_probs"].append((runner, store_prefix)) or {f"{store_prefix}log_probs": [1]}
        ),
    )
    monkeypatch.setattr(
        base_module,
        "run_optimizer_steps",
        lambda args, rid, it, n, runner, *, profiler: calls["steps"].append(runner),
    )
    monkeypatch.setattr(base_module, "compute_advantages_and_returns", lambda args, data: data.update(adv=True))
    monkeypatch.setattr(base_module, "log_rollout_data", lambda rid, args, data: None)
    monkeypatch.setattr(base_module, "timer", _noop_timer)
    monkeypatch.setattr(base_module, "inverse_timer", _noop_timer)
    return calls


def test_the_rollout_step_runs_ref_then_actor_then_optimizer_under_the_right_stages(loop):
    stages: list = []
    actor = _Provider(stages, with_ref=True)
    rollout_data = {}

    actor._train_core(rollout_id=3, rollout_data=rollout_data)

    assert stages == [
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
    assert loop["log_probs"] == [(actor.ref_runner, "ref_"), (actor.runner, "")]
    assert loop["steps"] == [actor.runner]
    assert rollout_data == {"ref_log_probs": [1], "log_probs": [1], "adv": True}
    assert actor.after_rollout_calls == [3]
    actor.prof.step.assert_called_once_with(rollout_id=3)
    actor.routing_replay.fill.assert_called_once()
    assert actor.routing_replay.fill.call_args.kwargs["align"] is actor.align_token_side_channel


def test_without_a_reference_model_there_is_no_ref_pass(loop):
    stages: list = []
    actor = _Provider(stages, with_ref=False)
    actor._train_core(rollout_id=0, rollout_data={})
    assert loop["log_probs"] == [(actor.runner, "")]
    assert "enter:fallthrough" not in stages


def test_train_returns_a_normal_output_and_logs_perf(loop, monkeypatch):
    actor = _Provider([], with_ref=False)
    actor._train_core = MagicMock()
    monkeypatch.setattr(base_module, "get_rollout_data", lambda args, ref, **kw: ({"tokens": []}, nullcontext()))
    with patch(f"{_MODULE}.train_metric_utils.log_perf_data_raw") as perf, patch(f"{_MODULE}.dist") as dist:
        dist.get_rank.return_value = 0
        result = actor.train(5, object())
    assert result == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_called_once()
    perf.assert_called_once()
    assert perf.call_args.kwargs["compute_total_fwd_flops"] is None


def test_debug_rollout_only_trains_nothing(loop, monkeypatch):
    actor = _Provider([], with_ref=False)
    actor.args.debug_rollout_only = True
    actor._train_core = MagicMock()
    monkeypatch.setattr(base_module, "get_rollout_data", lambda args, ref, **kw: ({}, nullcontext()))
    assert actor.train(0, object()) == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_not_called()


def _weight_actor(*, ci_test):
    actor = object.__new__(TorchNativeTrainRayActor)
    actor.args = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, ci_test=ci_test)
    actor.weight_updater = MagicMock(weight_version=4)
    return actor


def test_update_weights_reconnects_syncs_and_checks_the_version_in_order():
    actor = _weight_actor(ci_test=True)
    info = SimpleNamespace(rollout_engines=[object()])
    with patch(f"{_MODULE}.clear_memory"), patch(f"{_MODULE}.print_memory"):
        assert actor.update_weights(info) == 4
    assert [c[0] for c in actor.weight_updater.method_calls] == [
        "reconnect_if_needed",
        "update_weights",
        "verify_engine_version",
    ]


def test_the_version_check_is_ci_only():
    actor = _weight_actor(ci_test=False)
    with patch(f"{_MODULE}.clear_memory"), patch(f"{_MODULE}.print_memory"):
        actor.update_weights(SimpleNamespace(rollout_engines=[object()]))
    actor.weight_updater.verify_engine_version.assert_not_called()


def test_sleep_and_wake_up_move_the_provider_surface_only_under_offload_train():
    actor = _Provider([], with_ref=False)
    with patch(f"{_MODULE}.offload_to_host") as off, patch(f"{_MODULE}.reload_to_device") as on:
        actor.sleep()
        actor.wake_up()
        off.assert_not_called()
        on.assert_not_called()
        actor.args.offload_train = True
        actor.sleep()
        actor.wake_up()
        off.assert_called_once_with(actor.model_parts, actor.optimizers)
        on.assert_called_once_with(actor.model_parts, actor.optimizers)
