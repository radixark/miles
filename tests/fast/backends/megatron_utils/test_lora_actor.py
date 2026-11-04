import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _actor_with_sampler_export(monkeypatch, write_sampler_export):
    for name, attributes in {
        "miles.backends.megatron_utils.actor": {"MegatronTrainRayActor": object},
        "miles.backends.megatron_utils.lora": {"checkpoint": None, "model": None},
        "miles.backends.megatron_utils.lora.optimizer": {"SlotOptimizer": object},
        "miles.backends.training_utils.checkpoint.adapter.adapter_dir": {"write_sampler_export": write_sampler_export},
        "miles.backends.training_utils.checkpoint.adapter.commit": {"UncommittedAdapterDir": object},
        "miles.backends.training_utils.data.rollout": {"get_rollout_data": None},
        "miles.backends.training_utils.parallel": {"get_parallel_state": None},
        "miles.utils.object_store": {"StoreObjectRef": object},
        "miles.utils.tracking_utils.structured_log": {"with_logs": lambda fn: fn},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/lora/actor.py"
    spec = importlib.util.spec_from_file_location("lora_actor_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor = module.MultiLoRATrainRayActor()
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    return actor


class _Slots:
    targets = object()

    def parametrization(self, **fields):
        return SimpleNamespace(**fields)

    def lora_rank(self, slot):
        return 8


def test_export_slot_writes_the_slots_sampler_export_under_the_gateways_parametrization(monkeypatch):
    calls = []
    actor = _actor_with_sampler_export(monkeypatch, lambda **kwargs: calls.append(kwargs) or "written")
    actor.adapter_slots = _Slots()
    actor.args = SimpleNamespace(lora_dropout=0.1)

    assert actor.export_slot(3, "/root", "v1", "digest", {"lora_rank": 8, "base_model": "m"}) == "written"

    (call,) = calls
    assert call == dict(
        root=Path("/root"),
        name="v1",
        members=actor.adapter_slots,
        slot=3,
        parametrization=SimpleNamespace(lora_rank=8, base_model="m"),
        targets=_Slots.targets,
        tenant_digest="digest",
        lora_dropout=0.1,
    )
    with pytest.raises(AssertionError, match="disagree on rank"):
        actor.export_slot(3, "/root", "v2", "digest", {"lora_rank": 4, "base_model": "m"})
