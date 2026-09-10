"""The shared RL step of the torch-native backends, driven through a fake provider.

Checks call counts and ordering rather than numerics: the bugs this code is
prone to are structural (a missing zero_grad between optimizer steps, a step
applied per microbatch instead of per step, a pass run under the wrong replay
stage), and those are invisible in a loss curve.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils import torch_native_actor as base_module
from miles.backends.training_utils.step_runner import LinearStepRunner, StepMetrics
from miles.backends.training_utils.torch_native_actor import TorchNativeTrainRayActor

_MODULE = "miles.backends.training_utils.torch_native_actor"


@contextmanager
def _recording_context(stages, name):
    stages.append(f"enter:{name}")
    yield
    stages.append(f"exit:{name}")


def _fake_replay(stages: list):
    return SimpleNamespace(
        FALLTHROUGH="fallthrough",
        RECORD="record",
        REPLAY_FORWARD="replay_forward",
        REPLAY_BACKWARD="replay_backward",
        stage=lambda name: _recording_context(stages, name),
        fill=MagicMock(side_effect=lambda *a, **k: stages.append("fill")),
        log_prob_stage=lambda args: "replay_forward",
        rewind=MagicMock(side_effect=lambda: stages.append("rewind")),
        reset=MagicMock(side_effect=lambda: stages.append("reset")),
    )


class _Profiler:
    def iterate_train_actor(self, it):
        return it

    def iterate_train_log_probs(self, it):
        return it

    step = MagicMock()


class _Provider(TorchNativeTrainRayActor):
    def __init__(self, stages=None, *, with_ref=False):
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
        self.routing_replay = _fake_replay(stages if stages is not None else [])
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
def _noop_timer(_name):
    yield


class _DataIterator:
    def __init__(self):
        self.resets = 0
        self.fetches = 0

    def reset(self):
        self.resets += 1
        return self


# --- the rollout step ---------------------------------------------------------------


@pytest.fixture
def step(monkeypatch):
    calls = {"log_probs": [], "steps": []}
    iterator = object()
    monkeypatch.setattr(base_module, "get_data_iterator", lambda args, parts, data: ([iterator], [2]))
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
    monkeypatch.setattr(base_module, "inverse_timer", _noop_timer)
    return calls


def test_the_rollout_step_runs_ref_then_actor_then_optimizer_under_the_right_stages(step):
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
    assert step["log_probs"] == [(actor.ref_runner, "ref_"), (actor.runner, "")]
    assert step["steps"] == [actor.runner]
    assert rollout_data == {"ref_log_probs": [1], "log_probs": [1], "adv": True}
    assert actor.after_rollout_calls == [3]
    actor.prof.step.assert_called_once_with(rollout_id=3)
    assert actor.routing_replay.fill.call_args.kwargs["align"] is actor.align_token_side_channel


def test_without_a_reference_model_there_is_no_ref_pass(step):
    stages: list = []
    actor = _Provider(stages, with_ref=False)
    actor._train_core(rollout_id=0, rollout_data={})
    assert step["log_probs"] == [(actor.runner, "")]
    assert "enter:fallthrough" not in stages


def test_train_returns_a_normal_output_and_logs_perf(step, monkeypatch):
    actor = _Provider()
    actor._train_core = MagicMock()
    monkeypatch.setattr(base_module, "get_rollout_data", lambda args, ref, **kw: ({"tokens": []}, nullcontext()))
    with patch(f"{_MODULE}.train_metric_utils.log_perf_data_raw") as perf, patch(f"{_MODULE}.dist") as dist:
        dist.get_rank.return_value = 0
        result = actor.train(5, object())
    assert result == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_called_once()
    perf.assert_called_once()
    assert perf.call_args.kwargs["compute_total_fwd_flops"] is None


def test_debug_rollout_only_trains_nothing(step, monkeypatch):
    actor = _Provider()
    actor.args.debug_rollout_only = True
    actor._train_core = MagicMock()
    monkeypatch.setattr(base_module, "get_rollout_data", lambda args, ref, **kw: ({}, nullcontext()))
    assert actor.train(0, object()) == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_not_called()


# --- the loops -----------------------------------------------------------------------


@pytest.fixture
def loops():
    """Patch out everything that needs real tensors, keeping the control flow."""
    calls: list[str] = []

    def fake_get_batch(data_iterator, keys, *a, **kw):
        data_iterator.fetches += 1
        calls.append("batch")
        return {"unconcat_tokens": None, "total_lengths": [1], "response_lengths": [1]}

    def fake_loss(args, batch, num_microbatches, logits, apply_megatron_loss_scaling):
        calls.append("loss")
        assert apply_megatron_loss_scaling is False, "the shared loop never goes through a PP schedule"
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


def _runner(calls):
    def forward(batch):
        calls.append("forward")
        return torch.zeros(1, requires_grad=True)

    return LinearStepRunner(
        forward,
        lambda: calls.append("zero_grad"),
        lambda: (calls.append("step"), StepMetrics(grad_norm=0.5))[1],
    )


def _run_steps(calls, num_microbatches, runner=None):
    data_iterator = _DataIterator()
    _Provider()._optimizer_steps(runner or _runner(calls), data_iterator, num_microbatches, rollout_id=0)
    return data_iterator


def test_gradients_are_cleared_once_per_optimizer_step(loops):
    _run_steps(loops, [2, 3])
    assert loops.count("zero_grad") == 2


def test_zero_grad_precedes_the_microbatches_of_its_step(loops):
    _run_steps(loops, [2])
    assert loops.index("zero_grad") < loops.index("forward")


def test_optimizer_steps_once_per_step_not_per_microbatch(loops):
    _run_steps(loops, [4])
    assert loops.count("forward") == 4
    assert loops.count("step") == 1


def test_each_step_logs_exactly_once(loops):
    _run_steps(loops, [1, 1, 1])
    assert [c for c in loops if c.startswith("log:")] == ["log:0", "log:1", "log:2"]


def test_grad_norm_check_only_under_ci_test(loops):
    _run_steps(loops, [1])
    assert "check_grad_norm" not in loops


def test_the_iterator_is_rewound_before_the_loop(loops):
    data_iterator = _run_steps(loops, [2, 2])
    assert data_iterator.resets == 1
    assert data_iterator.fetches == 4


def test_fetch_and_compute_stay_interleaved_for_a_linear_runner(loops):
    """Microbatches reach the runner as a generator: a linear runner pulls one,
    computes on it, then pulls the next. Materializing them up front is the
    schedule-owning runners' choice, not the loop's."""
    _run_steps(loops, [3])
    assert [c for c in loops if c in ("batch", "forward")] == ["batch", "forward"] * 3


def test_the_runner_gets_one_forward_backward_call_per_optimizer_step(loops):
    """The seam is per optimizer step (pytorch/torchtitan#3856): a PP schedule
    needs every microbatch of the step in one call."""
    step_calls = []
    runner = _runner(loops)
    inner = runner.forward_backward_step
    runner.forward_backward_step = lambda batches, closure: step_calls.append(1) or inner(batches, closure)
    _run_steps(loops, [2, 3], runner=runner)
    assert len(step_calls) == 2


def test_log_probs_collects_one_entry_per_microbatch(loops):
    data_iterator = _DataIterator()
    result = _Provider()._log_probs(LinearStepRunner(lambda batch: torch.zeros(1)), data_iterator, [2, 3])
    assert result == {"n": 5}
    assert data_iterator.resets == 1


def test_log_probs_skips_entropy_for_a_prefixed_pass(loops):
    """Only the actor pass feeds entropy to the loss hub; ref/teacher passes do not."""
    seen = {}

    def spy(**kwargs):
        seen["with_entropy"] = kwargs["with_entropy"]
        return {"log_probs": 1}

    with patch.object(base_module, "get_log_probs_and_entropy", spy):
        _Provider()._log_probs(LinearStepRunner(lambda b: torch.zeros(1)), _DataIterator(), [1], "ref_")
    assert seen["with_entropy"] is False


def test_a_forward_only_runner_refuses_to_train():
    runner = LinearStepRunner(lambda batch: torch.zeros(1))
    with pytest.raises(RuntimeError, match="forward passes only"):
        runner.zero_grad()


# --- weights and offload -------------------------------------------------------------


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
        move_state.assert_called_with(actor.optimizers, "cuda")
