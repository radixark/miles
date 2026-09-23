import json
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.fsdp_utils import checkpoint
from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput


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


class _ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self):
        return SimpleNamespace(logits=self.weight)


@pytest.fixture
def make_fsdp_actor(monkeypatch):
    """Exercise the real training loop, backward and optimizer with CPU tensors."""
    monkeypatch.setattr(
        actor_module, "get_data_iterator", lambda _args, _model, data: ([Mock()], data["num_microbatches"])
    )
    monkeypatch.setattr(actor_module, "get_batch", Mock(return_value={}))
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", Mock())
    monkeypatch.setattr(actor_module, "log_rollout_data", Mock())
    monkeypatch.setattr(actor_module, "aggregate_train_losses", Mock(return_value={}))
    monkeypatch.setattr(actor_module, "log_train_step", Mock())
    monkeypatch.setattr(actor_module, "timer", _noop_timer)
    monkeypatch.setattr(actor_module, "precision_forward_context", lambda _policy: nullcontext())
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    for name in ("fill", "rewind", "reset"):
        monkeypatch.setattr(actor_module.routing_replay, name, Mock())
    monkeypatch.setattr(actor_module.routing_replay, "stage", lambda _stage: nullcontext())
    monkeypatch.setattr(actor_module.routing_replay, "log_prob_stage", Mock(return_value="log_probs"))
    monkeypatch.setattr(
        actor_module,
        "loss_function",
        lambda *, logits, num_microbatches, **_kwargs: (logits.square() / num_microbatches, None, {}),
    )

    clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def clip_cpu_grad_norm(parameters, max_norm):
        norm = clip_grad_norm(parameters, max_norm)
        assert torch.isfinite(norm) and norm > 0
        return SimpleNamespace(full_tensor=lambda: norm)

    monkeypatch.setattr(actor_module.torch.nn.utils, "clip_grad_norm_", clip_cpu_grad_norm)

    def make_actor():
        actor = object.__new__(actor_module.FSDPTrainRayActor)
        actor.args = Namespace(
            data_pad_size_multiplier=1,
            qkv_format="thd",
            clip_grad=1.0,
            ci_test=False,
            save_debug_train_data=None,
            ref_update_interval=None,
            start_rollout_id=0,
            no_load_rng=True,
        )
        actor.model = _ScalarModel()
        actor.optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.01)
        actor.lr_scheduler = Mock()
        actor.ref_model = None
        actor.precision_policy = None
        actor.prof = SimpleNamespace(iterate_train_actor=iter, step=Mock())
        actor._compute_log_prob = Mock(return_value={})
        actor._get_model_inputs_args = Mock(return_value={})
        actor.global_step = 0
        actor.micro_step = 0
        return actor

    return make_actor


@pytest.fixture
def checkpoint_io(monkeypatch, tmp_path):
    """Keep metadata and tracker I/O real; stub distributed tensor transport and CUDA."""
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(checkpoint.dist, "barrier", Mock())
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(checkpoint.torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(checkpoint.dcp, "save", Mock())
    monkeypatch.setattr(checkpoint.dcp, "load", Mock())
    return tmp_path


@pytest.mark.parametrize("num_microbatches", [[1, 1, 1], [2, 1, 3]])
def test_fsdp_training_counters_follow_completed_work(make_fsdp_actor, num_microbatches):
    actor = make_fsdp_actor()

    for completed_rollouts, rollout_id in enumerate((7, 8), start=1):
        actor._train_core(rollout_id, {"num_microbatches": num_microbatches})

        expected_steps = completed_rollouts * len(num_microbatches)
        assert actor.optimizer.state[actor.model.weight]["step"].item() == expected_steps
        assert actor.model.weight.item() < 1.0
        assert actor.global_step == expected_steps
        assert actor.micro_step == completed_rollouts * sum(num_microbatches)


def test_fsdp_failed_optimizer_step_only_counts_completed_microbatches(make_fsdp_actor, monkeypatch):
    actor = make_fsdp_actor()
    monkeypatch.setattr(actor.optimizer, "step", Mock(side_effect=RuntimeError("optimizer failed")))

    with pytest.raises(RuntimeError, match="optimizer failed"):
        actor._train_core(0, {"num_microbatches": [2]})

    assert actor.global_step == 0
    assert actor.micro_step == 2
    actor.lr_scheduler.step.assert_not_called()


def test_fsdp_failed_backward_does_not_count_a_microbatch(make_fsdp_actor, monkeypatch):
    actor = make_fsdp_actor()
    actor._train_step({}, step_id=0, num_microbatches=2)
    monkeypatch.setattr(actor_module, "loss_function", Mock(return_value=(torch.tensor(1.0), None, {})))

    with pytest.raises(RuntimeError, match="does not require grad"):
        actor._train_step({}, step_id=0, num_microbatches=2)

    assert actor.global_step == 0
    assert actor.micro_step == 1


def test_fsdp_training_counters_survive_checkpoint_resume(make_fsdp_actor, checkpoint_io):
    actor = make_fsdp_actor()
    actor.args.save = str(checkpoint_io)
    actor._train_core(2, {"num_microbatches": [2, 1, 3]})

    checkpoint.save(actor, iteration=2)

    metadata = json.loads((checkpoint_io / "iter_0000003" / "meta.json").read_text())
    assert metadata["global_step"] == 3
    assert metadata["micro_step"] == 6
    assert metadata["step_counters_version"] == 1
    assert (metadata["iteration"], metadata["rollout_id"], metadata["next_rollout_id"]) == (3, 2, 3)

    resumed = make_fsdp_actor()
    resumed.args.load = str(checkpoint_io)
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))

    assert (resumed.global_step, resumed.micro_step) == (3, 6)
    assert resumed.args.start_rollout_id == 3
    resumed._train_core(resumed.args.start_rollout_id, {"num_microbatches": [4, 2]})
    assert (resumed.global_step, resumed.micro_step) == (5, 12)


@pytest.mark.parametrize("legacy_metadata", [True, False], ids=["legacy-zeros", "missing-metadata"])
def test_fsdp_legacy_progress_stays_unknown_after_training_and_resave(make_fsdp_actor, checkpoint_io, legacy_metadata):
    checkpoint_dir = checkpoint_io / "iter_0000003"
    (checkpoint_dir / "model").mkdir(parents=True)
    (checkpoint_io / "latest_checkpointed_iteration.txt").write_text("3")
    if legacy_metadata:
        (checkpoint_dir / "meta.json").write_text(
            json.dumps({"iteration": 3, "rollout_id": 2, "next_rollout_id": 3, "global_step": 0, "micro_step": 0})
        )

    actor = make_fsdp_actor()
    actor.args.load = actor.args.save = str(checkpoint_io)
    actor.args.start_rollout_id = None
    checkpoint.finalize_load(actor, checkpoint.load(actor))

    assert (actor.global_step, actor.micro_step) == (None, None)
    assert actor.args.start_rollout_id == 3
    actor._train_core(3, {"num_microbatches": [2]})
    assert actor.optimizer.state[actor.model.weight]["step"].item() == 1
    assert (actor.global_step, actor.micro_step) == (None, None)
    checkpoint.save(actor, iteration=3)

    metadata = json.loads((checkpoint_io / "iter_0000004" / "meta.json").read_text())
    assert metadata["global_step"] is None
    assert metadata["micro_step"] is None
    resumed = make_fsdp_actor()
    resumed.args.load = str(checkpoint_io)
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    assert (resumed.global_step, resumed.micro_step) == (None, None)
    assert resumed.args.start_rollout_id == 4
