"""The torchtitan multi-LoRA actor's slot RPCs in the order the Tinker gateway issues them."""

from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=10, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# the package applies the torch 2.14 backports torchtitan needs at import, so it must load first
import miles.backends.torchtitan_utils  # noqa: F401

# isort: split
from miles.backends.torchtitan_utils import lora_actor
from miles.backends.torchtitan_utils.lora_checkpoint import TorchtitanAdapterSlots

_SOURCE = {"path": "/ckpt", "load_optimizer": True, "parametrization_fields": {}}


class _RecordedSlots(TorchtitanAdapterSlots):
    """``AdapterSlots``' own claim bookkeeping over recorded backend draws."""

    def __init__(self, events: list):
        self._multi_lora, self.num_slots, self._claimed_slots, self.targets = True, 3, set(), None
        self._events, self.head_rows = events, None

    def _init_slot(self, slot, lora_rank, lora_alpha, seed):
        self._events.append(("init", slot, lora_alpha))

    def _clear_slot(self, slot):
        self._events.append(("clear", slot))

    def slot_params(self, slot):
        return [torch.nn.Parameter(torch.zeros(1))]

    def parametrization(self, **fields):
        return None


@pytest.fixture
def actor_and_events(monkeypatch):
    events = []
    monkeypatch.setattr(
        lora_actor, "TorchtitanSlotOptimizer", lambda slot, params: SimpleNamespace(slot=slot, zero_grads=lambda: None)
    )
    monkeypatch.setattr(lora_actor, "load_adapter_source", lambda path, *args, **kwargs: events.append(("load", path)))
    actor = lora_actor.TorchtitanMultiLoRATrainRayActor.__new__(lora_actor.TorchtitanMultiLoRATrainRayActor)
    actor.adapter_slots, actor.slot_optimizers = _RecordedSlots(events), {}
    actor._checkpoint_members = lambda: None
    return actor, events


def test_load_state_into_a_created_model_claims_its_slot_again_at_the_sources_alpha(actor_and_events, monkeypatch):
    """create_model claims the slot; load_state then loads into that same live slot."""
    actor, events = actor_and_events
    monkeypatch.setattr(actor, "_check_load_source", lambda *args: (None, None), raising=False)

    assert actor.load_slot(1, 8, 16.0) is None
    assert actor.load_slot(1, 8, 32.0, source=_SOURCE) is None
    assert events == [("init", 1, 16.0), ("clear", 1), ("init", 1, 32.0), ("load", Path("/ckpt"))]

    actor.unload_slot(1)
    assert 1 not in actor.slot_optimizers
    assert actor.load_slot(1, 8, 16.0) is None, "another tenant creates a model on the freed slot"


def test_a_refused_source_leaves_the_live_slot_and_its_optimizer_alone(actor_and_events, monkeypatch):
    actor, events = actor_and_events
    monkeypatch.setattr(actor, "_check_load_source", lambda *args: ("saved with another layout", None), raising=False)
    actor.load_slot(1, 8, 16.0)
    optimizer = actor.slot_optimizers[1]

    refused = actor.load_slot(1, 8, 32.0, source=_SOURCE)

    assert refused == {"error": "saved with another layout", "error_category": "user"}
    assert events == [("init", 1, 16.0)] and actor.slot_optimizers[1] is optimizer


def test_creating_a_model_on_a_claimed_slot_fails(actor_and_events):
    actor, _ = actor_and_events
    actor.load_slot(1, 8, 16.0)

    with pytest.raises(AssertionError, match="slot 1 is already claimed"):
        actor.load_slot(1, 8, 16.0)


def test_the_multi_lora_actor_is_selected_for_torchtitan():
    from miles.ray.specs.train import _MULTI_LORA_ACTOR_CLASSES
    from miles.utils.function_registry import load_function

    assert load_function(_MULTI_LORA_ACTOR_CLASSES["torchtitan"]) is lora_actor.TorchtitanMultiLoRATrainRayActor


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
