"""serve_tinker startup: an explicit slot count launches everything at once; ``auto``
probes a one-slot trainer, resolves the count, and rebuilds before the engines launch."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import serve_tinker


@pytest.mark.parametrize("requested", [-1, 4], ids=["auto", "explicit"])
async def test_engines_launch_with_the_resolved_slot_count(monkeypatch, requested):
    events = []
    args = Namespace(
        multi_lora=True,
        multi_lora_n_adapters=requested,
        tinker_checkpoint_root="/ckpt",
        tinker_base_model=None,
        hf_checkpoint="model",
        target_modules=["q_proj", "gate_proj"],
        lora_rank=32,
        lora_alpha=64,
        tinker_server_port=10613,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sglang_router_ip=None,
        sglang_router_port=None,
    )

    class Trainer:
        def __init__(self, **kwargs):
            self.slots = kwargs["args"].multi_lora_n_adapters

        async def init(self):
            events.append(("trainer", self.slots))

        async def dispose(self):
            events.append(("dispose", self.slots))

    async def restart(specs):
        events.append(("restart", specs))

    def launch(namespace, *, trainer_only):
        events.append(("launch", namespace.multi_lora_n_adapters, trainer_only))
        return SimpleNamespace(restart_with_specs=SimpleNamespace(remote=restart))

    async def init_engines():
        # the router address is known once the engines are up
        args.sglang_router_ip, args.sglang_router_port = "10.0.0.1", 30000
        events.append(("engines", args.multi_lora_n_adapters))

    backends, configs = [], []
    monkeypatch.setattr(serve_tinker, "configure_logger", lambda *a, **kw: None)
    monkeypatch.setattr(serve_tinker, "init_http_client", lambda *a, **kw: None)
    monkeypatch.setattr(serve_tinker, "MainProcessIdentity", lambda: None)
    monkeypatch.setattr(serve_tinker.object_store, "init_instance", lambda *a, **kw: None)
    monkeypatch.setattr(serve_tinker, "launch_worker_manager", launch)
    monkeypatch.setattr(serve_tinker, "compute_specs", lambda namespace: ("specs", namespace.multi_lora_n_adapters))
    monkeypatch.setattr(serve_tinker, "InferenceController", lambda _: SimpleNamespace(init=init_engines))
    monkeypatch.setattr(serve_tinker, "TrainerController", Trainer)
    monkeypatch.setattr(
        serve_tinker,
        "MilesBackend",
        lambda trainer, router_url, dp_size: backends.append((trainer.slots, router_url, dp_size)) or object(),
    )
    monkeypatch.setattr(serve_tinker, "probe_slot_capacity", AsyncMock(return_value=["probe"]))
    monkeypatch.setattr(serve_tinker, "resolve_slot_capacity", lambda namespace, probes: 3)
    monkeypatch.setattr(serve_tinker, "convert_target_modules_to_megatron", lambda names: ["linear_qkv", "linear_fc1"])
    monkeypatch.setattr(
        serve_tinker,
        "TinkerService",
        lambda backend, config: configs.append(config) or SimpleNamespace(run=AsyncMock()),
    )
    monkeypatch.setattr(serve_tinker, "build_app", lambda service: None)
    monkeypatch.setattr(serve_tinker.uvicorn, "Config", lambda *a, **kw: None)
    monkeypatch.setattr(serve_tinker.uvicorn, "Server", lambda config: SimpleNamespace(serve=AsyncMock()))

    await serve_tinker.serve(args)

    resolved = 3 if requested == -1 else 4
    assert events[-2:] == [("engines", resolved), ("trainer", resolved)]
    assert configs[0].n_slots == resolved
    assert backends[-1] == (resolved, "http://10.0.0.1:30000", 2)
    if requested == -1:
        assert events[:4] == [("launch", 1, True), ("trainer", 1), ("dispose", 1), ("restart", ("specs", 3))]
        assert backends[0] == (1, "", 2)  # the probe backend never reaches a router
    else:
        assert events[0] == ("launch", 4, False)
        assert len(backends) == 1
