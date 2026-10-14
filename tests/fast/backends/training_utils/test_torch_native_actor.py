from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.training_utils.torch_native import actor as base_module
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native.offload import _map_optimizer_state

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
            use_sampling_support_replay=False,
        )
        self.model_parts = [MagicMock()]
        self.optimizers = [object()]
        self.hf_config = object()
        self.prof = _Profiler()
        self.runner = object()
        self.align_token_side_channel = lambda t, pad: t
        self._heartbeat = MagicMock()

    def _step_runner(self):
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
        lambda self, runner, it, n, rollout_id, num_rollouts=None: calls["steps"].append((runner, num_rollouts)),
    )
    monkeypatch.setattr(base_module, "compute_advantages_and_returns", lambda args, data: data.update(adv=True))
    monkeypatch.setattr(base_module, "log_rollout_data", lambda rid, args, data: None)
    monkeypatch.setattr(base_module, "timer", _noop_timer)
    return calls


def test_the_rollout_step_runs_ref_then_actor_then_optimizer_under_the_right_stages(step):
    actor = _Provider()
    actor.ref_runner = object()
    actor._ref_context = lambda: _recording(step["stages"], "ref")
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
    assert step["steps"] == [(actor.runner, None)]
    assert rollout_data == {"ref_log_probs": [1], "log_probs": [1], "adv": True}
    assert step["replay"].fill.call_args.kwargs["align"] is actor.align_token_side_channel


def test_moving_optimizer_state_does_not_grow_state_for_parameters_that_have_none():
    model = torch.nn.Linear(4, 4, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    assert model.bias not in optimizer.state

    _map_optimizer_state([optimizer], lambda tensor: tensor.to("cpu"))

    assert model.bias not in optimizer.state
    assert optimizer.state[model.weight]["exp_avg"].device.type == "cpu"


class _RecordingRunner:
    def __init__(self):
        self.closures = []

    def zero_grad(self):
        pass

    def forward_backward_step(self, batches, loss_closure):
        list(batches)
        self.closures.append(loss_closure)
        return []

    def apply_step(self):
        return SimpleNamespace(grad_norm=0.0, extra_metrics={})


@pytest.mark.parametrize("per_token", [True, False])
def test_each_optimizer_step_hands_its_global_token_count_to_the_loss(monkeypatch, per_token):
    """The count is taken once per step, before that step's micro-batches are fetched."""
    events = []
    counts = iter([torch.tensor(7), torch.tensor(11)])

    def _count(data_iterator, num_microbatches, *, group):
        events.append(("count", num_microbatches, group))
        return next(counts)

    monkeypatch.setattr(base_module, "_global_loss_tokens", _count)
    state = SimpleNamespace(intra_dp=SimpleNamespace(group="dp"), is_metrics_rank=False)
    monkeypatch.setattr(base_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(base_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(base_module, "log_train_step", lambda **kwargs: None)
    monkeypatch.setattr(base_module, "aggregate_train_losses", lambda losses, num_rollouts=None: {})
    monkeypatch.setattr(
        TorchNativeTrainRayActor,
        "_fetch_batches",
        lambda self, progress, it, keys: (events.append(("fetch",)) or [{} for _ in progress]),
    )
    actor = _Provider()
    actor.args.calculate_per_token_loss = per_token
    runner = _RecordingRunner()

    actor._optimizer_steps(runner, _DataIterator(), [2, 3], rollout_id=0)

    if per_token:
        assert events == [("count", 2, "dp"), ("fetch",), ("count", 3, "dp"), ("fetch",)]
        assert [closure.args[1:] for closure in runner.closures] == [(2, torch.tensor(7)), (3, torch.tensor(11))]
    else:
        assert events == [("fetch",), ("fetch",)]
        assert [closure.args[1:] for closure in runner.closures] == [(2, None), (3, None)]


def test_the_rollout_step_hands_the_stamped_rollout_counts_to_the_optimizer_steps(step):
    actor = _Provider()

    actor._train_core(rollout_id=3, rollout_data={"num_rollouts": [32]})

    assert step["steps"] == [(actor.runner, [32])]


@pytest.mark.parametrize("per_token", [True, False])
def test_each_optimizer_step_hands_its_rollout_count_to_the_loss_and_the_metrics(monkeypatch, per_token):
    """A stamped num_rollouts reaches the loss of its own step; the metrics divide by it only for the
    per-sample loss, as in the Megatron actor."""
    metric_counts = []
    state = SimpleNamespace(intra_dp=SimpleNamespace(group="dp"), is_metrics_rank=False)
    monkeypatch.setattr(base_module, "_global_loss_tokens", lambda *args, **kwargs: torch.tensor(1))
    monkeypatch.setattr(base_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(base_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(base_module, "log_train_step", lambda **kwargs: None)
    monkeypatch.setattr(
        base_module,
        "aggregate_train_losses",
        lambda losses, num_rollouts=None: metric_counts.append(num_rollouts) or {},
    )
    monkeypatch.setattr(
        TorchNativeTrainRayActor, "_fetch_batches", lambda self, progress, it, keys: [{} for _ in progress]
    )
    actor = _Provider()
    actor.args.calculate_per_token_loss = per_token
    runner = _RecordingRunner()

    actor._optimizer_steps(runner, _DataIterator(), [2, 3], rollout_id=0, num_rollouts=[5, 6])

    assert [closure.keywords for closure in runner.closures] == [{"num_rollouts": 5}, {"num_rollouts": 6}]
    assert metric_counts == ([None, None] if per_token else [5, 6])


def test_a_rollout_count_per_optimizer_step_is_required():
    with pytest.raises(AssertionError, match="one rollout count per optimizer step"):
        _Provider()._optimizer_steps(_RecordingRunner(), _DataIterator(), [2, 3], rollout_id=0, num_rollouts=[5])


def test_the_train_keys_carry_the_rollout_mask_sums():
    """Without them every sample of a split rollout is its own token mean."""
    assert "rollout_mask_sums" in base_module.TRAIN_KEYS
