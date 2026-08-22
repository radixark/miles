import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.api_backends.multi_lora.checkpoint as tc
from miles.backends.megatron_utils.api_backends.multi_lora.checkpoint import (
    FORMAT,
    find_slot_state,
    stable_slot_param_name,
)


class TestStableName:
    def test_strips_exactly_the_target_slot(self):
        # load_adapter consumes ".adapter." keys; a co-tenant's index must
        # survive untouched, including prefix-colliding double-digit slots.
        name = "decoder.layers.0.self_attention.linear_qkv.adapters.3.linear_in.weight"
        assert stable_slot_param_name(name, 3) == "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"
        assert stable_slot_param_name(name, 2) == name
        assert ".adapter." in stable_slot_param_name("m.adapters.0.linear_out.weight", 0)
        assert stable_slot_param_name("m.adapters.12.linear_in.weight", 12) == "m.adapter.linear_in.weight"
        assert stable_slot_param_name("m.adapters.12.linear_in.weight", 1) == "m.adapters.12.linear_in.weight"


def make_adapter(tmp_path, name="a", rank=8, alpha=16):
    config = SimpleNamespace(save=tmp_path, rank=rank, alpha=alpha)
    return SimpleNamespace(name=name, registration_id="r1", slot=0, step=3, version=2, config=config)


def write_manifest(base, **overrides):
    manifest = {"format": FORMAT, "name": "a", "rank_lora": 8, "alpha": 16, "optimizer_step": 3, "world_size": 1}
    manifest.update(overrides)
    base.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, base / "manifest.pt")


class TestManifestGating:
    """State compatibility is fenced by format, topology, rank, and alpha, never display name."""

    def test_missing_dir_or_manifest_means_no_state(self, tmp_path):
        assert find_slot_state(SimpleNamespace(config=SimpleNamespace(save=None))) is None
        adapter = make_adapter(tmp_path)
        (tmp_path / "slot_state").mkdir()
        assert find_slot_state(adapter) is None

    def test_foreign_name_is_loadable_but_foreign_shape_is_not(self, tmp_path):
        adapter = make_adapter(tmp_path)
        base = tmp_path / "slot_state"
        write_manifest(base, name="someone-else")
        assert find_slot_state(adapter) == base

        write_manifest(base, rank_lora=4)
        assert find_slot_state(adapter) is None

        write_manifest(base, world_size=8)
        assert find_slot_state(adapter) is None

        write_manifest(base, format="something-old")
        assert find_slot_state(adapter) is None


class TestSlotStateRoundTrip:
    """Cross-slot restore requires matching ownership and save generation before mutation."""

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
            self.moment.copy_(state["optimizer"]["state"][0]["exp_avg"])
            for group, saved in zip(self.param_groups, state["optimizer"]["param_groups"], strict=True):
                group.update({key: value for key, value in saved.items() if key != "params"})

    def _round_trip(self, tmp_path, monkeypatch, target_children, ttl_seconds=None, after_save=None):
        adapter = make_adapter(tmp_path)
        adapter.step = 7

        source = [self._FakeChild(slot=0, moment=1.5)]
        source[0].param_groups[0]["step"] = 7
        children_by_slot = {0: source, 1: target_children}
        monkeypatch.setattr(tc, "_slot_children", lambda optimizer, slot: children_by_slot[slot])
        monkeypatch.setattr(
            tc,
            "named_adapter_slot_parameters",
            lambda model, slot: iter([("m.adapter.linear_in.weight", torch.ones(2))]),
        )
        bridge = ModuleType("megatron.bridge.peft.multi_lora_layers")
        loads: dict = {}
        bridge.load_adapter = lambda model, slot, weights: loads.update(weights=weights) or len(weights)
        bridge.init_adapter_slot = lambda model, slot, rank, alpha: loads.update(rank=rank, alpha=alpha)
        monkeypatch.setitem(sys.modules, "megatron.bridge.peft.multi_lora_layers", bridge)

        tc.save_slot_state(
            args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter, reason="state", ttl_seconds=ttl_seconds
        )
        if after_save is not None:
            after_save()
        adapter.slot = 1
        step = tc.load_slot_state(args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter)
        return step, loads, adapter

    def test_optimizer_state_restores_into_another_slot(self, tmp_path, monkeypatch):
        target = [self._FakeChild(slot=1, moment=0.0)]
        step, loads, _ = self._round_trip(tmp_path, monkeypatch, target)
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

    def test_torn_save_is_refused(self, tmp_path, monkeypatch):
        # An interrupted overwrite leaves shards of one save under the
        # manifest of another; the shared save token catches the mix.
        def cross_generation_manifest():
            manifest_path = tmp_path / "slot_state" / "manifest.pt"
            manifest = torch.load(manifest_path, weights_only=True)
            manifest["save_id"] = "another-generation"
            torch.save(manifest, manifest_path)

        target = [self._FakeChild(slot=1, moment=0.0)]
        with pytest.raises(ValueError, match="torn"):
            self._round_trip(tmp_path, monkeypatch, target, after_save=cross_generation_manifest)

    def test_ownership_signature_mismatch_is_refused_before_mutation(self, tmp_path, monkeypatch):
        # Positional optimizer entries follow LayerWise DP ownership: when the
        # target slot's rank owns DIFFERENT parameters, a blind positional load
        # would silently restore the wrong state — refuse, weights untouched.
        adapter = make_adapter(tmp_path)
        param_a, param_b = torch.zeros(1), torch.zeros(1)

        def child_with(param, slot):
            child = self._FakeChild(slot=slot, moment=0.0)
            child.param_groups[0]["params"] = [param]
            return child

        children_by_slot = {0: [child_with(param_a, 0)], 1: [child_with(param_b, 1)]}
        names_by_slot = {
            0: [("m.adapter.linear_in.weight", param_a)],
            1: [("m.adapter.linear_out.weight", param_b)],
        }
        monkeypatch.setattr(tc, "_slot_children", lambda optimizer, slot: children_by_slot[slot])
        monkeypatch.setattr(tc, "named_adapter_slot_parameters", lambda model, slot: iter(names_by_slot[slot]))
        bridge = ModuleType("megatron.bridge.peft.multi_lora_layers")
        loads: dict = {}
        bridge.load_adapter = lambda model, slot, weights: loads.update(weights=weights) or len(weights)
        bridge.init_adapter_slot = lambda model, slot, rank, alpha: loads.update(rank=rank, alpha=alpha)
        monkeypatch.setitem(sys.modules, "megatron.bridge.peft.multi_lora_layers", bridge)

        tc.save_slot_state(args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter, reason="state")
        adapter.slot = 1
        with pytest.raises(ValueError, match="ownership"):
            tc.load_slot_state(args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter)
        assert loads == {}  # refused before any weight or optimizer mutation

    def test_ttl_is_recorded_in_the_manifest(self, tmp_path, monkeypatch):
        target = [self._FakeChild(slot=1, moment=0.0)]
        self._round_trip(tmp_path, monkeypatch, target, ttl_seconds=3600)
        manifest = torch.load(tmp_path / "slot_state" / "manifest.pt", weights_only=True)
        assert manifest["ttl_seconds"] == 3600
