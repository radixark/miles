"""Exercise startup ordering across the driver/worker serialization boundary."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [-1, 4], ids=["auto", "explicit"])
async def test_engines_receive_the_resolved_count(monkeypatch, tmp_path, requested):
    import serve_tinker as gateway

    events = []
    args = Namespace(
        multi_lora=True,
        multi_lora_n_adapters=requested,
        tinker_checkpoint_root=str(tmp_path),
        tinker_base_model=None,
        hf_checkpoint="model",
        target_modules=["linear_qkv", "linear_fc1"],
        lora_alpha=32,
        tinker_server_port=10639,
        max_tokens_per_gpu=8192,
        train_memory_margin_bytes=0,
        engine_host_lora_budget_bytes=None,
        sglang_router_ip=None,
        sglang_router_port=None,
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )

    class Trainer:
        def __init__(self, **kwargs):
            self.count = kwargs["args"].multi_lora_n_adapters

        async def init(self):
            events.append(("trainer", self.count))

        async def dispose(self):
            events.append(("dispose", self.count))

    async def restart(specs):
        events.append(("restart", specs))

    manager = SimpleNamespace(restart_with_specs=SimpleNamespace(remote=restart))

    def launch(namespace, *, trainer_only):
        events.append(("launch", namespace.multi_lora_n_adapters, trainer_only))
        return manager

    async def init_inference():
        args.sglang_router_ip, args.sglang_router_port = "10.0.0.1", 8123

    inference = SimpleNamespace(init=AsyncMock(side_effect=init_inference))
    for name in ("configure_logger", "init_http_client"):
        monkeypatch.setattr(gateway, name, lambda *a, **kw: None)
    monkeypatch.setattr(gateway.object_store, "init_instance", lambda *a, **kw: None)
    monkeypatch.setattr(gateway, "MainProcessIdentity", lambda: None)
    monkeypatch.setattr(gateway, "launch_worker_manager", launch)
    monkeypatch.setattr(gateway, "TrainerController", Trainer)
    monkeypatch.setattr(gateway, "InferenceController", lambda _: inference)
    urls = []

    def backend(trainer, router_url, *, dp_size):
        assert dp_size == 8
        urls.append(router_url)
        return object()

    monkeypatch.setattr(gateway, "MilesBackend", backend)
    probe = AsyncMock(return_value=[])
    monkeypatch.setattr(gateway, "probe_slot_capacity", probe)
    monkeypatch.setattr(gateway, "resolve_slot_capacity", lambda *a: 3)
    monkeypatch.setattr(gateway, "compute_specs", lambda namespace: namespace.multi_lora_n_adapters)
    monkeypatch.setattr(gateway, "TinkerService", lambda *a: SimpleNamespace(run=AsyncMock()))
    monkeypatch.setattr(gateway, "build_app", lambda _: None)
    monkeypatch.setattr(gateway.uvicorn, "Config", lambda *a, **kw: None)
    monkeypatch.setattr(gateway.uvicorn, "Server", lambda _: SimpleNamespace(serve=AsyncMock()))

    await gateway.serve(args)
    inference.init.assert_awaited_once()
    assert urls[-1] == "http://10.0.0.1:8123"
    if requested == -1:
        assert events == [("launch", 1, True), ("trainer", 1), ("dispose", 1), ("restart", 3), ("trainer", 3)]
        assert args.sglang_max_loaded_loras == 6
        assert (tmp_path / "slot-capacity.json").is_file()
        probe.assert_awaited_once()
    else:
        assert events == [("launch", 4, False), ("trainer", 4)]
        probe.assert_not_awaited()
        assert not (tmp_path / "slot-capacity.json").exists()
