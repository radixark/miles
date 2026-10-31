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
        "miles.backends.training_utils.data.rollout": {
            "get_rollout_data": lambda *args, **kwargs: ({}, nullcontext())
        },
        "miles.backends.training_utils.parallel": {
            "get_parallel_state": lambda: SimpleNamespace(train_parallel_config=lambda **kwargs: None)
        },
        "miles.utils.lora.utils": {"AdapterSpec": object},
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
