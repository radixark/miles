"""Named Ray actor wrapping the multi-LoRA backend + HTTP server."""

import time
from functools import cache
from typing import Any

import ray

from miles.utils.adapter_config import AdapterRun
from miles.utils.misc import SingletonMeta, get_current_node_ip, load_function
from miles.utils.multi_lora import MultiLoRABackend, MultiLoRAHTTPServer
from miles.utils.ray_utils import compute_ray_pin_head_options

CONTROLLER_NAME = "miles_multi_lora_controller"
CONTROLLER_NAMESPACE = "miles"


@cache
def get_multi_lora_controller():
    return ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)


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
                self.snapshot = await get_multi_lora_controller().snapshot.remote()
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


@ray.remote(num_cpus=0)
class MultiLoRAController:
    def __init__(self, args, router_url: str, host: str = "0.0.0.0") -> None:
        backend_cls = _load_subclass(getattr(args, "multi_lora_backend_path", None), MultiLoRABackend)
        server_cls = _load_subclass(getattr(args, "multi_lora_http_server_path", None), MultiLoRAHTTPServer)
        self.backend = backend_cls(args, router_url)
        self.server = server_cls(self.backend, host, api_port=getattr(args, "multi_lora_api_port", 0))

    async def start(self) -> int:
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


def create_controller(args, router_url: str, host: str = "0.0.0.0"):
    # Pinned to the head node so the API sits at a port-forwardable address.
    return MultiLoRAController.options(
        name=CONTROLLER_NAME,
        namespace=CONTROLLER_NAMESPACE,
        **compute_ray_pin_head_options(),
    ).remote(args, router_url, host)
