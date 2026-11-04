"""The multi-LoRA actor's slot RPCs in the order the Tinker gateway issues them — no GPU."""

from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.lora import actor as actor_module
from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots


def _actor(monkeypatch):
    """The actor over real ``AdapterSlots`` claim bookkeeping; the backend's draws and optimizers are recorded."""
    events = []
    slots = BridgeAdapterSlots.__new__(BridgeAdapterSlots)
    slots._multi_lora, slots.num_slots, slots._claimed_slots, slots.targets = True, 3, set(), None
    monkeypatch.setattr(
        BridgeAdapterSlots, "_init_slot", lambda self, slot, rank, alpha, seed: events.append(("init", slot, alpha))
    )
    monkeypatch.setattr(BridgeAdapterSlots, "_clear_slot", lambda self, slot: events.append(("clear", slot)))
    monkeypatch.setattr(BridgeAdapterSlots, "slot_parameters", lambda self, slot: [torch.nn.Parameter(torch.zeros(1))])
    monkeypatch.setattr(
        lora_model,
        "SlotOptimizer",
        lambda args, model, slot, params: SimpleNamespace(slot=slot, zero_grads=lambda: None),
    )
    monkeypatch.setattr(lora_checkpoint, "find_load_mismatch", lambda *args, **kwargs: None)
    monkeypatch.setattr(lora_checkpoint, "load_slot", lambda *args, **kwargs: events.append(("load", args[3])))
    monkeypatch.setattr(BridgeAdapterSlots, "parametrization", lambda self, **fields: None)
    actor = actor_module.MultiLoRATrainRayActor.__new__(actor_module.MultiLoRATrainRayActor)
    actor.args, actor.model, actor.adapter_slots, actor.slot_optimizers = None, [], slots, {}
    actor._checkpoint_members = lambda: None
    return actor, events


def test_load_state_into_a_created_model_claims_its_slot_again_at_the_sources_alpha(monkeypatch):
    """create_model claims the slot; load_state then loads into that same live slot."""
    actor, events = _actor(monkeypatch)

    assert actor.load_slot(1, 8, 16.0) is None
    assert actor.load_slot(1, 8, 32.0, ckpt_path="/ckpt", parametrization_fields={}) is None

    assert events == [("init", 1, 16.0), ("clear", 1), ("init", 1, 32.0), ("load", "/ckpt")]
    assert actor.adapter_slots._claimed_slots == {1}
