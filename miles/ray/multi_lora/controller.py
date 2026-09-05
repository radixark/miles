"""Worker wrapping the multi-LoRA backend + HTTP server."""

import time
from collections.abc import Sequence
from functools import cache
from typing import Any

from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.ray.multi_lora.http_server import MultiLoRAHTTPServer
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.specs.multi_lora import create_multi_lora_controller_handle
from miles.utils.adapter_config import AdapterRun
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import SingletonMeta, get_current_node_ip
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


@cache
def get_multi_lora_controller() -> BaseWorkerHandle:
    # TODO inject the factory instead of assuming the ray backend
    capability = RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())
    return create_multi_lora_controller_handle(capability=capability)


class AdaptersCache(metaclass=SingletonMeta):
    """TTL-cached controller snapshot; get/get_all expose the sampleable
    projection (active + retiring)."""

    def __init__(self, ttl_s: float = 1.0) -> None:
        self.ttl_s = ttl_s
        self.snapshot: dict = {"pending": {}, "active": {}, "retiring": {}, "cleanup": []}
        self.last_refresh: float | None = None

    async def get_snapshot(self) -> dict:
        now = time.monotonic()
        if self.last_refresh is None or now - self.last_refresh >= self.ttl_s:
            try:
                self.snapshot = await get_multi_lora_controller().snapshot()
                self.last_refresh = now
            except Exception:
                pass
        return self.snapshot

    async def get_all(self) -> dict[str, "AdapterRun"]:
        snapshot = await self.get_snapshot()
        return {**snapshot["active"], **snapshot["retiring"]}

    async def get(self, adapter_name: str) -> "AdapterRun | None":
        return (await self.get_all()).get(adapter_name)


def _load_subclass(path: str | None, base_cls):
    if not path:
        return base_cls
    cls = load_function(path)
    assert issubclass(cls, base_cls), f"{path} must point to a {base_cls.__name__} subclass, got {cls}"
    return cls


class MultiLoRAController:
    def __init__(self, *, args, router_providers: Sequence[BaseWorkerProvider], host: str = "0.0.0.0") -> None:
        configure_logger(args, source=SimpleProcessIdentity(component="multi_lora_controller"))

        self.args = args
        self._router_providers = router_providers
        self.host = host
        self.backend: MultiLoRABackend | None = None
        self.server: MultiLoRAHTTPServer | None = None
        self._init_once = InitOnce(type(self).__name__)

    @init_once
    async def init(self) -> int:
        args = self.args
        await resolve_router_addrs(args, router_providers=self._router_providers)
        router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

        backend_cls = _load_subclass(getattr(args, "multi_lora_backend_path", None), MultiLoRABackend)
        server_cls = _load_subclass(getattr(args, "multi_lora_http_server_path", None), MultiLoRAHTTPServer)
        self.backend = backend_cls(args, router_url)
        self.server = server_cls(self.backend, self.host, api_port=getattr(args, "multi_lora_api_port", 0))

        await self.backend.init()
        await self.server.start()
        return self.server.actual_api_port

    async def stop(self) -> None:
        await self.server.stop()
        await self.backend.close()

    async def register_adapter(self, name: str, config: Any) -> dict:
        return await self.backend.register(name, config)

    async def deregister_adapter(self, name: str) -> None:
        await self.backend.deregister(name)

    async def retire_adapters(self) -> list[str]:
        return await self.backend.retire_adapters()

    async def free_slot(self, name: str) -> int:
        return await self.backend.free_slot(name)

    def record_weight_update(self, names: list[str]) -> None:
        self.backend.registry.record_weight_update(names)

    def record_batch_adapters(self, rollout_id: int, groups: dict[str, int], step_names: list[str]) -> None:
        self.backend.registry.record_batch_adapters(rollout_id, groups, step_names)

    def mark_batch_trained(self, rollout_id: int) -> list[str]:
        return self.backend.registry.mark_batch_trained(rollout_id)

    def resolve_num_step(self, name: str, dataset_rows: int) -> None:
        self.backend.registry.resolve_num_step(name, dataset_rows)

    def set_adapter_step(self, name: str, step: int) -> None:
        self.backend.registry.set_step(name, step)

    def adapter_step(self, name: str) -> int:
        return self.backend.registry.step_count(name)

    def snapshot(self) -> dict:
        return self.backend.registry.snapshot()

    def http_host(self) -> str:
        return get_current_node_ip()

    def api_port(self) -> int:
        return self.server.actual_api_port
