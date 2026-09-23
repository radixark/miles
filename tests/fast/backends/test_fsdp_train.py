from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock

import torch

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.parallel import GroupInfo, set_parallel_state
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state


@contextmanager
def _noop_timer(_name: str) -> Iterator[None]:
    yield


def test_fsdp_train_debug_rollout_only_returns_a_normal_output(monkeypatch):
    """A debug-rollout-only FSDP step trains nothing yet answers the driver with a NORMAL output."""
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = Namespace(offload_train=False, debug_rollout_only=True)
    actor._heartbeat = Mock()
    actor._train_core = Mock()
    actor.wake_up = Mock()
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    monkeypatch.setattr(actor_module, "timer", _noop_timer)
    monkeypatch.setattr(actor_module, "inverse_timer", _noop_timer)

    result = actor.train(3, object())

    assert result == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_not_called()


def test_train_step_scales_per_token_loss_before_backward(monkeypatch):
    """The pre-scanned scale multiplies the loss before backward, so it reaches the gradients."""
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = Namespace(calculate_per_token_loss=True)
    actor.model = Mock()
    actor.precision_policy = Mock()
    monkeypatch.setattr(actor_module, "routing_replay", Mock(stage=lambda _name: nullcontext()))
    monkeypatch.setattr(actor_module, "precision_forward_context", lambda _policy: nullcontext())

    param = torch.nn.Parameter(torch.tensor(3.0))

    def fake_loss_function(**_kwargs):
        return param * 2.0, torch.tensor(1), {"keys": [], "values": torch.tensor([])}

    monkeypatch.setattr(actor_module, "loss_function", fake_loss_function)
    batch = {"tokens": torch.zeros(1, 1, dtype=torch.long), "position_ids": torch.zeros(1, 1, dtype=torch.long)}

    actor._train_step(batch=batch, step_id=0, num_microbatches=1, per_token_loss_scale=torch.tensor(2.0))
    assert param.grad.item() == 4.0

    param.grad = None
    actor._train_step(batch=batch, step_id=0, num_microbatches=1)
    assert param.grad.item() == 2.0


def _make_train_core_actor(monkeypatch, args, rollout_data, num_microbatches):
    """An FSDPTrainRayActor whose _train_core surroundings are stubbed down to the training loop."""
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = args
    actor.ref_model = None
    actor.model = Mock(parameters=lambda: iter([]))
    actor.optimizer = Mock(param_groups=[])
    actor.lr_scheduler = Mock()
    actor.prof = Mock()
    actor.prof.iterate_train_actor = lambda it: it
    actor._compute_log_prob = Mock(return_value={})

    state = make_parallel_state()
    state.intra_dp = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(state)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: None)

    data_iterator = DataIterator(rollout_data, micro_batch_size=2)
    monkeypatch.setattr(
        actor_module, "get_data_iterator", lambda *_args, **_kwargs: ([data_iterator], num_microbatches)
    )
    monkeypatch.setattr(actor_module, "get_batch", lambda *_args, **_kwargs: {"tokens": None})
    monkeypatch.setattr(
        actor_module,
        "routing_replay",
        Mock(stage=lambda _name: nullcontext(), fill=Mock(), rewind=Mock(), reset=Mock()),
    )
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", Mock())
    monkeypatch.setattr(actor_module, "log_rollout_data", Mock())
    monkeypatch.setattr(actor_module, "aggregate_train_losses", Mock(return_value={}))
    monkeypatch.setattr(actor_module, "log_train_step", Mock())
    monkeypatch.setattr(actor_module, "timer", _noop_timer)
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda _params, _clip: Mock(full_tensor=lambda: torch.tensor(1.0))
    )

    return actor


def test_train_core_pre_scans_scales_after_reset_and_groups_them_by_step(monkeypatch):
    """Per-token mode scans the reset schedule once; every micro-batch of a step shares its scale."""
    rollout_data = {
        "loss_masks": [torch.full((n,), 1, dtype=torch.int) for n in (4, 6, 3, 5, 2, 7)],
    }
    args = Namespace(
        calculate_per_token_loss=True,
        data_pad_size_multiplier=128,
        qkv_format="thd",
        clip_grad=1.0,
        ci_test=False,
        save_debug_train_data=None,
        ref_update_interval=None,
    )
    actor = _make_train_core_actor(monkeypatch, args, rollout_data, num_microbatches=[2, 1])

    scan_offsets = []
    real_scan = actor_module.get_per_token_loss_scales

    def scanning(data_iterator, num_microbatches):
        scan_offsets.append(data_iterator.offset)
        return real_scan(data_iterator, num_microbatches)

    monkeypatch.setattr(actor_module, "get_per_token_loss_scales", scanning)
    train_step_calls = []
    actor._train_step = lambda **kwargs: train_step_calls.append(kwargs) or {"keys": [], "values": torch.tensor([])}

    actor._train_core(rollout_id=0, rollout_data=dict(rollout_data))

    assert scan_offsets == [0]
    assert [call["step_id"] for call in train_step_calls] == [0, 0, 1]
    # step 0 covers 4+6+3+5 tokens, step 1 covers 2+7; dp_size is 1
    scales = torch.stack([call["per_token_loss_scale"] for call in train_step_calls])
    assert torch.allclose(scales, torch.tensor([1 / 18, 1 / 18, 1 / 9]))


def test_train_core_skips_the_scan_for_sample_mean(monkeypatch):
    """Sample-mean mode keeps the legacy behavior: no scan, no scale passed to _train_step."""
    rollout_data = {
        "loss_masks": [torch.full((n,), 1, dtype=torch.int) for n in (4, 6, 3, 5, 2, 7)],
    }
    args = Namespace(
        calculate_per_token_loss=False,
        data_pad_size_multiplier=128,
        qkv_format="thd",
        clip_grad=1.0,
        ci_test=False,
        save_debug_train_data=None,
        ref_update_interval=None,
    )
    actor = _make_train_core_actor(monkeypatch, args, rollout_data, num_microbatches=[2, 1])

    scan = Mock()
    monkeypatch.setattr(actor_module, "get_per_token_loss_scales", scan)
    train_step_calls = []
    actor._train_step = lambda **kwargs: train_step_calls.append(kwargs) or {"keys": [], "values": torch.tensor([])}

    actor._train_core(rollout_id=0, rollout_data=dict(rollout_data))

    scan.assert_not_called()
    assert all(call["per_token_loss_scale"] is None for call in train_step_calls)
