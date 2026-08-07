"""Slot-state sidecar: stable naming, manifest gating, and slot round-trip.

The stable name must strip EXACTLY the target slot's index — stripping a
co-tenant's would let one adapter's sidecar overwrite another slot's weights
on load."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.multi_lora_utils.checkpoint import FORMAT, find_slot_state, stable_slot_param_name


class TestStableName:
    def test_strips_exactly_the_target_slot(self):
        # load_adapter consumes ".adapter." keys (the expose_adapter_slot
        # export layout); a co-tenant's index must survive untouched, including
        # prefix-colliding double-digit slots.
        name = "decoder.layers.0.self_attention.linear_qkv.adapters.3.linear_in.weight"
        assert stable_slot_param_name(name, 3) == "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"
        assert stable_slot_param_name(name, 2) == name
        assert ".adapter." in stable_slot_param_name("m.adapters.0.linear_out.weight", 0)
        assert stable_slot_param_name("m.adapters.12.linear_in.weight", 12) == "m.adapter.linear_in.weight"
        assert stable_slot_param_name("m.adapters.12.linear_in.weight", 1) == "m.adapters.12.linear_in.weight"


class TestManifestGating:
    def _adapter(self, tmp_path, name="a"):
        config = SimpleNamespace(save=tmp_path, rank=8, alpha=16)
        return SimpleNamespace(name=name, registration_id="r1", slot=0, step=3, version=2, config=config)

    def test_sidecar_discovery_gates(self, tmp_path):
        # No save dir, no manifest, or a foreign-name manifest all mean "no
        # sidecar"; a matching manifest resolves to its directory.
        assert find_slot_state(SimpleNamespace(config=SimpleNamespace(save=None))) is None

        adapter = self._adapter(tmp_path)
        base = tmp_path / "slot_state"
        base.mkdir()
        assert find_slot_state(adapter) is None  # dir exists, no manifest

        torch.save(
            {"format": FORMAT, "name": "someone-else", "optimizer_step": 3, "world_size": 1},
            base / "manifest.pt",
        )
        assert find_slot_state(adapter) is None  # foreign manifest

        torch.save(
            {"format": FORMAT, "name": "a", "optimizer_step": 3, "world_size": 1},
            base / "manifest.pt",
        )
        assert find_slot_state(adapter) == base


class TestSidecarRoundTrip:
    """A sidecar saved from slot A must restore positionally into slot B,
    re-stamping the slot tag (the save carries the SOURCE slot's), and a
    child-count mismatch must be refused outright, never partially loaded."""

    class _FakeChild:
        def __init__(self, slot: int, moment: float):
            self.param_groups = [{"params": [0], "miles_multi_lora_slot": slot, "step": 0}]
            self.moment = torch.full((2,), moment)

        def state_dict(self):
            return {
                "optimizer": {
                    "state": {0: {"exp_avg": self.moment.clone()}},
                    "param_groups": [dict(group) for group in self.param_groups],
                }
            }

        def load_state_dict(self, state):
            # Mirrors torch/MCore semantics: copy state tensors in place, take
            # ALL non-params group keys — including foreign slot tags — from
            # the save.
            self.moment.copy_(state["optimizer"]["state"][0]["exp_avg"])
            for group, saved in zip(self.param_groups, state["optimizer"]["param_groups"], strict=True):
                group.update({key: value for key, value in saved.items() if key != "params"})

    def _round_trip(self, tmp_path, monkeypatch, target_children):
        import miles.backends.megatron_utils.multi_lora_utils.checkpoint as mlc
        import miles.backends.megatron_utils.multi_lora_utils.optimizer as mlo

        config = SimpleNamespace(save=tmp_path, rank=8, alpha=16)
        adapter = SimpleNamespace(name="a", registration_id="r1", slot=0, step=7, version=2, config=config)

        source = [self._FakeChild(slot=0, moment=1.5)]
        source[0].param_groups[0]["step"] = 7
        children_by_slot = {0: source, 1: target_children}
        monkeypatch.setattr(mlo, "_slot_children", lambda optimizer, slot: children_by_slot[slot])
        monkeypatch.setattr(
            mlc,
            "named_adapter_slot_parameters",
            lambda model, slot: iter([("m.adapter.linear_in.weight", torch.ones(2))]),
        )
        bridge = ModuleType("megatron.bridge.peft.multi_lora_layers")
        loads: dict = {}
        bridge.load_adapter = lambda model, slot, weights: loads.update(weights=weights) or len(weights)
        bridge.init_adapter_slot = lambda model, slot, rank, alpha: loads.update(rank=rank, alpha=alpha)
        monkeypatch.setitem(sys.modules, "megatron.bridge.peft.multi_lora_layers", bridge)

        mlc.save_slot_state(args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter, reason="swap")
        adapter.slot = 1
        step = mlc.load_slot_state(args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter)
        return step, loads

    def test_optimizer_state_restores_into_another_slot(self, tmp_path, monkeypatch):
        target = [self._FakeChild(slot=1, moment=0.0)]
        step, loads = self._round_trip(tmp_path, monkeypatch, target)
        assert step == 7
        assert loads["rank"] == 8 and loads["alpha"] == 16
        assert torch.equal(loads["weights"]["m.adapter.linear_in.weight"], torch.ones(2))
        assert torch.equal(target[0].moment, torch.full((2,), 1.5))
        group = target[0].param_groups[0]
        assert group["step"] == 7
        assert group["miles_multi_lora_slot"] == 1  # re-stamped over the saved slot-0 tag

    def test_child_count_mismatch_is_refused(self, tmp_path, monkeypatch):
        two_children = [self._FakeChild(slot=1, moment=0.0), self._FakeChild(slot=1, moment=0.0)]
        with pytest.raises(ValueError, match="refusing partial restore"):
            self._round_trip(tmp_path, monkeypatch, two_children)


class TestSwapInSidecarSentinel:
    """A sidecar that exists with optimizer step 0 (an adapter swapped out
    before its first step) must be restored as-is — only a MISSING sidecar
    (None) may fall back to the weights-only registration re-init."""

    def _swap_in_with(self, monkeypatch, load_result):
        import miles.backends.megatron_utils.multi_lora_utils.checkpoint as mlc
        import miles.backends.megatron_utils.multi_lora_utils.optimizer as mlo
        import miles.backends.megatron_utils.multi_lora_utils.scheduler as mls
        import miles.backends.megatron_utils.multi_lora_utils.utils as mlu

        calls = []
        monkeypatch.setattr(mlc, "load_slot_state", lambda *a, **k: load_result)
        monkeypatch.setattr(mlu, "_register_adapter", lambda *a, **k: calls.append("register") or 0)
        monkeypatch.setattr(mlo, "reload_adapter_slot_model_params", lambda *a, **k: calls.append("reload"))
        monkeypatch.setattr(mls, "install_slot_scheduler", lambda *a, **k: calls.append("scheduler"))
        adapter = SimpleNamespace(name="a", slot=0)
        step = mlc.swap_in(args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter)
        return step, calls

    def test_step_zero_sidecar_is_not_reinitialized(self, monkeypatch):
        step, calls = self._swap_in_with(monkeypatch, load_result=0)
        assert step == 0
        assert calls == ["scheduler"]  # no register, no master re-derivation

    def test_missing_sidecar_falls_back_to_registration(self, monkeypatch):
        step, calls = self._swap_in_with(monkeypatch, load_result=None)
        assert step == 0
        assert calls == ["register", "reload", "scheduler"]
