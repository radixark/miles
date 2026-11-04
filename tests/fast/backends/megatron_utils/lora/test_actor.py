"""The multi-LoRA actor's slot RPCs in the order the Tinker gateway issues them — no GPU."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora import actor as actor_module
from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots


class _RecordedSlots(BridgeAdapterSlots):
    """``AdapterSlots``' own claim bookkeeping over recorded backend draws."""

    def __init__(self, events: list):
        self._multi_lora, self.num_slots, self._claimed_slots, self.targets = True, 3, set(), None
        self._events = events

    def _init_slot(self, slot, lora_rank, lora_alpha, seed):
        self._events.append(("init", slot, lora_alpha))

    def _clear_slot(self, slot):
        self._events.append(("clear", slot))

    def slot_parameters(self, slot):
        return [torch.nn.Parameter(torch.zeros(1))]

    def parametrization(self, **fields):
        return None


@pytest.fixture
def actor_and_events(monkeypatch):
    events = []
    monkeypatch.setattr(
        lora_model,
        "SlotOptimizer",
        lambda args, model, slot, params: SimpleNamespace(slot=slot, zero_grads=lambda: None),
    )
    monkeypatch.setattr(lora_checkpoint, "load_slot", lambda *args, **kwargs: events.append(("load", args[3])))
    actor = actor_module.MultiLoRATrainRayActor.__new__(actor_module.MultiLoRATrainRayActor)
    actor.args, actor.model, actor.adapter_slots, actor.slot_optimizers = None, [], _RecordedSlots(events), {}
    actor._checkpoint_members = lambda: None
    return actor, events


def test_load_state_into_a_created_model_claims_its_slot_again_at_the_sources_alpha(actor_and_events, monkeypatch):
    """create_model claims the slot; load_state then loads into that same live slot."""
    actor, events = actor_and_events
    monkeypatch.setattr(lora_checkpoint, "find_load_mismatch", lambda *args, **kwargs: None)

    assert actor.load_slot(1, 8, 16.0) is None
    assert actor.load_slot(1, 8, 32.0, ckpt_path="/ckpt", parametrization_fields={}) is None
    assert events == [("init", 1, 16.0), ("clear", 1), ("init", 1, 32.0), ("load", "/ckpt")]

    actor.unload_slot(1)
    assert actor.adapter_slots._claimed_slots == set() and actor.slot_optimizers == {}


def test_a_refused_source_leaves_the_live_slot_and_its_optimizer_alone(actor_and_events, monkeypatch):
    actor, events = actor_and_events
    monkeypatch.setattr(lora_checkpoint, "find_load_mismatch", lambda *args, **kwargs: "saved with another layout")
    actor.load_slot(1, 8, 16.0)
    optimizer = actor.slot_optimizers[1]

    refused = actor.load_slot(1, 8, 32.0, ckpt_path="/ckpt", parametrization_fields={})

    assert refused == {"error": "saved with another layout", "error_category": "user"}
    assert events == [("init", 1, 16.0)] and actor.slot_optimizers[1] is optimizer


def test_creating_a_model_on_a_claimed_slot_fails(actor_and_events):
    actor, _ = actor_and_events
    actor.load_slot(1, 8, 16.0)

    with pytest.raises(AssertionError, match="slot 1 is already claimed"):
        actor.load_slot(1, 8, 16.0)
