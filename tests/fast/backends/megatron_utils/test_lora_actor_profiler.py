import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.utils.profile_utils import TrainProfiler


@pytest.mark.parametrize("method", ["forward_backward", "forward_only"])
def test_lora_pass_emits_scheduled_trace(monkeypatch, method):
    model = SimpleNamespace(run_forward_backward=lambda *args, **kwargs: torch.ones(2).sum())
    dependencies = {
        "miles.backends.megatron_utils.actor": {"MegatronTrainRayActor": object},
        "miles.backends.megatron_utils.lora": {"checkpoint": None, "model": model},
        "miles.backends.megatron_utils.lora.optimizer": {"SlotOptimizer": object},
        "miles.backends.training_utils.checkpoint.adapter.adapter_dir": {"write_sampler_export": None},
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
    actor = module.MultiLoRATrainRayActor()
    actor.args = SimpleNamespace()
    actor.output_store_read = None
    actor.model = []
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor.prof = TrainProfiler(
        SimpleNamespace(backend=SimpleNamespace(use_pytorch_profiler=False, record_memory_history=False))
    )
    traces = []
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=lambda trace: traces.append(trace.events()),
    )
    actor.prof._torch_profiler_overall = profiler
    with profiler:
        for batch_id in range(2):
            assert getattr(actor, method)(batch_id, None).item() == 2
        assert len(traces) == 1
        assert any(event.name == "aten::sum" for event in traces[0])


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
