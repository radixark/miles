from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.torch_native import actor as base_module
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native.step_runner import LinearStepRunner, StepMetrics
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


def test_without_a_reference_model_there_is_no_ref_pass(step):
    actor = _Provider()
    actor._train_core(rollout_id=0, rollout_data={})
    assert step["log_probs"] == [(actor.runner, "")]
    assert "enter:fallthrough" not in step["stages"]


@pytest.fixture
def loops():
    calls: list[str] = []

    def fake_get_batch(data_iterator, keys, *a, **kw):
        data_iterator.fetches += 1
        calls.append("batch")
        return {"unconcat_tokens": None, "total_lengths": [1], "response_lengths": [1]}

    def fake_loss(args, batch, num_microbatches, logits, apply_megatron_loss_scaling):
        assert apply_megatron_loss_scaling is False
        return logits.sum(), 1, {}

    state = SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=0), is_metrics_rank=True)
    with (
        patch.object(base_module, "get_batch", fake_get_batch),
        patch.object(base_module, "loss_function", fake_loss),
        patch.object(base_module, "aggregate_train_losses", lambda x: {}),
        patch.object(base_module, "log_train_step", lambda **kw: calls.append(f"log:{kw['step_id']}")),
        patch.object(base_module, "check_grad_norm", lambda **kw: calls.append("check_grad_norm")),
        patch.object(base_module, "aggregate_forward_results", lambda store, *a, **kw: {"n": len(store)}),
        patch.object(base_module, "get_log_probs_and_entropy", lambda **kw: {"log_probs": 1, "entropy": 2}),
        patch.object(base_module, "get_parallel_state", lambda: state),
        patch.object(base_module, "timer", _noop_timer),
        patch.object(base_module.dist, "get_rank", lambda: 0),
    ):
        yield calls


def test_one_optimizer_step_per_schedule_entry(loops):
    def forward(batch):
        loops.append("forward")
        return torch.zeros(1, requires_grad=True)

    runner = LinearStepRunner(
        forward, lambda: loops.append("zero_grad"), lambda: (loops.append("step"), StepMetrics(grad_norm=0.5))[1]
    )
    step_calls = []
    inner = runner.forward_backward_step
    runner.forward_backward_step = lambda batches, closure: step_calls.append(1) or inner(batches, closure)
    data_iterator = _DataIterator()

    _Provider()._optimizer_steps(runner, data_iterator, [2, 3], rollout_id=0)

    assert loops.count("zero_grad") == 2
    assert loops.count("step") == 2
    assert loops.count("forward") == 5
    assert [c for c in loops if c.startswith("log:")] == ["log:0", "log:1"]
    assert "check_grad_norm" not in loops
    assert len(step_calls) == 2
    assert loops.index("zero_grad") < loops.index("forward")
    assert [c for c in loops if c in ("batch", "forward")] == ["batch", "forward"] * 5
    assert (data_iterator.resets, data_iterator.fetches) == (1, 5)


def test_log_probs_collects_one_entry_per_microbatch_and_only_the_actor_pass_asks_for_entropy(loops):
    seen = []

    def spy(**kwargs):
        seen.append(kwargs["with_entropy"])
        return {"log_probs": 1}

    data_iterator = _DataIterator()
    with patch.object(base_module, "get_log_probs_and_entropy", spy):
        actor = _Provider()
        result = actor._log_probs(LinearStepRunner(lambda batch: torch.zeros(1)), data_iterator, [2, 3])
        actor._log_probs(LinearStepRunner(lambda batch: torch.zeros(1)), _DataIterator(), [1], "ref_")
    assert result == {"n": 5}
    assert data_iterator.resets == 1
    assert seen == [True] * 5 + [False]


def test_update_weights_reconnects_syncs_and_checks_the_version_only_under_ci():
    actor = object.__new__(TorchNativeTrainRayActor)
    actor.args = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, ci_test=True)
    actor.weight_updater = MagicMock(weight_version=4)
    info = SimpleNamespace(rollout_engines=[object()])
    with patch(f"{_MODULE}.clear_memory"), patch(f"{_MODULE}.print_memory"):
        assert actor.update_weights(info) == 4
        actor.args.ci_test = False
        actor.weight_updater.reset_mock()
        actor.update_weights(info)
    assert [c[0] for c in actor.weight_updater.method_calls] == ["reconnect_if_needed", "update_weights"]


def test_sleep_and_wake_up_move_modules_and_optimizer_state_only_under_offload_train():
    actor = _Provider()
    (module,) = actor.model_parts
    with (
        patch(f"{_MODULE}.dist"),
        patch(f"{_MODULE}.get_gloo_group"),
        patch(f"{_MODULE}.clear_memory"),
        patch(f"{_MODULE}.print_memory"),
        patch(f"{_MODULE}.move_optimizer_state") as move_state,
    ):
        actor.sleep()
        actor.wake_up()
        module.to.assert_not_called()
        actor.args.offload_train = True
        actor.sleep()
        module.to.assert_called_once_with("cpu")
        move_state.assert_called_with(actor.optimizers, "cpu")
        actor.wake_up()
        module.to.assert_called_with("cuda")


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
