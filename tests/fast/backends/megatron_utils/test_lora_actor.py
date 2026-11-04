import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _lora_actor_module(monkeypatch, *, model=None, checkpoint=None):
    dependencies = {
        "miles.backends.megatron_utils.actor": {"MegatronTrainRayActor": object},
        "miles.backends.megatron_utils.lora": {"checkpoint": checkpoint, "model": model},
        "miles.backends.megatron_utils.lora.optimizer": {"SlotOptimizer": object},
        "miles.backends.training_utils.checkpoint.adapter.adapter_dir": {
            "write_external_export": None,
            "write_sampler_export": None,
        },
        "miles.backends.training_utils.checkpoint.adapter.commit": {"UncommittedAdapterDir": object},
        "miles.backends.training_utils.data.rollout": {
            "get_rollout_data": lambda *args, **kwargs: ({}, nullcontext())
        },
        "miles.backends.training_utils.parallel": {
            "get_parallel_state": lambda: SimpleNamespace(train_parallel_config=lambda **kwargs: None)
        },
        "miles.utils.object_store": {"StoreObjectRef": object},
        "miles.utils.tracking_utils.structured_log": {"with_logs": lambda fn: fn},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/lora/actor.py"
    spec = importlib.util.spec_from_file_location("lora_actor_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_source_the_slot_cannot_take_is_a_user_error_and_leaves_the_slot_alone(monkeypatch):
    loaded = []
    checkpoint = SimpleNamespace(
        find_load_mismatch=lambda *args, **shapes: "rank 4 != 8",
        load_slot=lambda *args: loaded.append("checkpoint"),
    )
    model = SimpleNamespace(load_slot=lambda *args: loaded.append("model"))
    module = _lora_actor_module(monkeypatch, model=model, checkpoint=checkpoint)
    actor = module.MultiLoRATrainRayActor()
    actor.args, actor.model, actor.slot_optimizers = None, [], {}
    actor.adapter_slots = SimpleNamespace(
        parametrization=lambda **fields: fields["base_model"],
        targets=None,
        global_shapes=lambda rank: {},
        head_rows=None,
    )

    source = {"path": "/ckpt", "load_optimizer": False, "parametrization_fields": {"base_model": "m"}}
    assert actor.load_slot(0, rank=8, alpha=16.0, source=source) == {"error": "rank 4 != 8", "error_category": "user"}
    assert loaded == [] and actor.slot_optimizers == {}


def _actor_with_sampler_export(monkeypatch, write_sampler_export):
    for name, attributes in {
        "miles.backends.megatron_utils.actor": {"MegatronTrainRayActor": object},
        "miles.backends.megatron_utils.lora": {"checkpoint": None, "model": None},
        "miles.backends.megatron_utils.lora.optimizer": {"SlotOptimizer": object},
        "miles.backends.training_utils.checkpoint.adapter.adapter_dir": {
            "write_external_export": None,
            "write_sampler_export": write_sampler_export,
        },
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
    )
    with pytest.raises(AssertionError, match="disagree on rank"):
        actor.export_slot(3, "/root", "v2", "digest", {"lora_rank": 4, "base_model": "m"})
