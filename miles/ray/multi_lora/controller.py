"""Ray actor for the Multi-LoRA operation control surface."""

from functools import cache
from typing import Any

import ray

from miles.ray.multi_lora.backend import MultiLoraOperationBackend
from miles.ray.multi_lora.http_server import AdapterRunControlServer
from miles.utils.misc import load_function
from miles.utils.ray_utils import compute_ray_pin_head_options

CONTROLLER_NAME = "miles_tinker_controller"
CONTROLLER_NAMESPACE = "miles"


@cache
def get_multi_lora_controller():
    return ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)


def _load_subclass(path: str | None, base_cls):
    if not path:
        return base_cls
    cls = load_function(path)
    assert issubclass(cls, base_cls), f"{path} must point to a {base_cls.__name__} subclass, got {cls}"
    return cls


@ray.remote(num_cpus=0)
class MultiLoraOperationController:
    # Loopback by default: the control plane executes client-referenced work
    # and must be fronted by the (future) authenticated tinker frontend.
    def __init__(self, args, router_url: str, host: str = "127.0.0.1") -> None:
        backend_cls = _load_subclass(getattr(args, "multi_lora_backend_path", None), MultiLoraOperationBackend)
        server_cls = _load_subclass(getattr(args, "multi_lora_http_server_path", None), AdapterRunControlServer)
        self.backend = backend_cls(args, router_url)
        self.server = server_cls(self.backend, host, api_port=getattr(args, "multi_lora_api_port", 0))

    async def start(self) -> int:
        await self.backend.init()
        await self.server.start()
        return self.server.actual_api_port

    async def stop(self) -> None:
        await self.server.stop()
        await self.backend.close()

    # ---------------- registration lifecycle ----------------

    async def register_adapter(self, name: str, config: Any) -> dict:
        return await self.backend.register(name, config)

    async def deregister_adapter(self, name: str, expected_registration_id: str | None = None) -> None:
        await self.backend.deregister(name, expected_registration_id)

    async def retire_adapters(self) -> list[str]:
        return await self.backend.retire_adapters()

    async def free_slot(self, name: str) -> int:
        return await self.backend.free_slot(name)

    def bootstrap_pending(self) -> list[str]:
        return self.backend.registry.bootstrap_pending()

    def mark_ready(self, names: list[str]) -> None:
        self.backend.registry.mark_ready(names)

    def record_weight_update(self, names: list[str]) -> None:
        self.backend.registry.record_weight_update(names)

    def set_trainer_ready(self) -> None:
        self.backend.mark_trainer_ready()

    def set_adapter_step(self, name: str, step: int) -> None:
        self.backend.set_adapter_step(name, step)

    def adapter_step(self, name: str) -> int:
        return self.backend.adapter_step(name)

    def snapshot(self) -> dict:
        return self.backend.registry.snapshot()

    # ---------------- operations ----------------

    def enqueue_operation(
        self,
        name: str,
        operation_id: str,
        ordinal: int,
        kind: str,
        payload: dict | None = None,
        expected_registration_id: str | None = None,
    ) -> dict:
        return self.backend.enqueue_operation(name, operation_id, ordinal, kind, payload, expected_registration_id)

    def claim_data_operation(self, name: str, registration_id: str) -> dict | None:
        # Claim-and-bind in this single actor call: no binding, no CLAIMED.
        return self.backend.claim_data_operation(name, registration_id)

    def acquire_batch_lease(self, bindings_by_operation: list):
        return self.backend.acquire_batch_lease(bindings_by_operation)

    def release_batch_lease(self, lease_metadata: dict) -> None:
        self.backend.release_batch_lease(lease_metadata)

    def claim_ready_control_operations(self) -> list[dict]:
        return self.backend.claim_ready_control_operations()

    def complete_control_operations(self, results: dict) -> None:
        self.backend.complete_control_operations(results)

    def commit_tinker_batch(self, accumulated: list, operation_ids: list, logprobs_by_op: dict | None = None) -> None:
        # ``accumulated`` is a list of exact (name, registration_id) keys;
        # normalize sequence types that crossed the Ray boundary.
        self.backend.commit_tinker_batch([tuple(key) for key in accumulated], list(operation_ids), logprobs_by_op)

    def fail_tinker_batch(self, operation_ids: list, error: str, lease_metadata: dict | None = None) -> None:
        # The abnormal-outcome finalizer for a dispatched data batch that did
        # not commit: still-CLAIMED operations terminal-fail typed server.
        self.backend.fail_tinker_batch(list(operation_ids), error, lease_metadata)

    def complete_operation(self, operation_id: str, result: dict | None = None) -> None:
        self.backend.operations.complete(operation_id, result)

    def fail_operation(self, operation_id: str, error: str, category: str = "server") -> None:
        self.backend.operations.fail(operation_id, error, category)

    def cancel_operation(self, operation_id: str) -> dict:
        return self.backend.operations.cancel(operation_id)

    def get_operation(self, operation_id: str) -> dict | None:
        return self.backend.operation_view(operation_id)

    def ack_operation(self, operation_id: str) -> None:
        self.backend.operations.ack(operation_id)

    def service_info(self) -> dict:
        return self.backend.service_info()

    def http_host(self) -> str:
        return self.server.advertised_host

    def api_port(self) -> int:
        return self.server.actual_api_port


def create_multi_lora_controller(args, router_url: str, host: str = "127.0.0.1"):
    # Pinned to the head node so the API sits at a port-forwardable address.
    return MultiLoraOperationController.options(
        name=CONTROLLER_NAME,
        namespace=CONTROLLER_NAMESPACE,
        **compute_ray_pin_head_options(),
    ).remote(args, router_url, host)
