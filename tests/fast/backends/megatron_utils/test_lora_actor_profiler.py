from types import SimpleNamespace

import pytest
import torch

from tests.fast.backends.megatron_utils.test_lora_actor import _lora_actor_module

from miles.utils.profile_utils import TrainProfiler


@pytest.mark.parametrize("method", ["forward_backward", "forward_only"])
def test_lora_pass_emits_scheduled_trace(monkeypatch, method):
    model = SimpleNamespace(run_forward_backward=lambda *args, **kwargs: torch.ones(2).sum())
    module = _lora_actor_module(monkeypatch, model=model)
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
