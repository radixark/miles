"""CPU proof through the gateway; all-rank CUDA/SGLang capture needs a GPU deployment."""

import json
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig, ModelRecord
from miles.tinker.server.app import build_app
from miles.utils.profile_utils import DynamicProfiler


class Backend:
    profiler = DynamicProfiler(external=(torch.nn.Linear(4, 4),))

    async def start_profile(self, target, config):
        return {"workers": [self.profiler.start(config)]}

    async def stop_profile(self, target):
        return {"workers": [self.profiler.stop()]}

    async def profiling_status(self, target):
        return {"workers": [self.profiler.status()]}


backend = Backend()
service = TinkerService(
    backend,
    GatewayConfig(
        base_model="local",
        n_slots=4,
        checkpoint_root=sys.argv[1],
        vocab_size=32,
        exports_tinker_names=False,
        profiling_enabled=True,
    ),
)
for slot, name in enumerate(("a", "b")):
    service.models[name] = ModelRecord(name, "session", "owner", slot, "local", 1, 2, "request")
with TestClient(build_app(service), headers={"X-API-Key": "owner"}) as client:
    target = {"model_id": "a", "target": "trainer"}
    options = {
        **target,
        "activities": ["CPU"],
        "wait": 1,
        "active": 1,
        "profile_memory": True,
        "profile_modules": True,
    }
    assert client.post("/api/v1/start_profiling", json=options).status_code == 200
    assert client.post("/api/v1/start_profiling", json={**options, "model_id": "b"}).status_code == 403
    backend.profiler.profile_step()
    with backend.profiler.capture(step=42):
        backend.profiler.external[0](torch.ones(2, 4)).sum().backward()
    backend.profiler.profile_step()
    status = client.get("/api/v1/profiling_status", params=target).json()["workers"][0]
    assert not status["active"] and status["captures"] == 1 and not status["error"], status
    manifest = Path(status["capture"])
    data = json.loads(manifest.read_text())
    assert data["step_start"] == data["step_end"] == 42 and data["operation_succeeded"]
    assert {"trace.json", "trace.modules.json", "trace.modules.txt"} <= {f["path"] for f in data["files"]}
    assert client.post("/api/v1/stop_profiling", json=target).status_code == 200
    assert client.post("/api/v1/start_profiling", json={**options, "model_id": "b"}).status_code == 200
print("PASS: multi-LoRA ownership, schedule, Perfetto trace, module memory, real step 42, restart")
