"""Operation-queue-owning backend: what is runnable per registration; mount via --multi-lora-backend-path miles.ray.multi_lora.operation_backend.MultiLoRAOperationBackend."""

import logging
from typing import Any

from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.ray.multi_lora.operations import CONTROL_KINDS, OperationQueue, OperationQueueSet
from miles.ray.multi_lora.registry import AdapterState

logger = logging.getLogger(__name__)


class MultiLoRAOperationBackend(MultiLoRABackend):
    """Registry-bound queue ownership: ACTIVE gating and slot enrichment over an OperationQueueSet."""

    def __init__(self, args: Any, router_url: str) -> None:
        super().__init__(args, router_url)
        self._operations = OperationQueueSet()

    def operation_queue(self, name: str) -> OperationQueue:
        return self._operations.get_or_create(name)

    async def register(self, name: str, config: Any) -> dict:
        result = await super().register(name, config)
        # A fresh queue per registration life: a re-registered name never sees its predecessor's ops.
        self._operations.replace(name)
        return result

    async def deregister(self, name: str) -> None:
        self._operations.fence(name, "registration retired before the operation ran")
        await super().deregister(name)

    # ------------------------------ operation rounds ------------------------------

    def collect_operation_round(self) -> dict:
        """Claim one round from ACTIVE registrations; slots come from the registry records."""
        eligible = {}
        for name in self._operations.queues:
            record = self.registry.find(name)
            if record is not None and record.state is AdapterState.ACTIVE:
                eligible[name] = record.slot
        data_ops: list[dict] = []
        control_ops: list[dict] = []
        for name, claimed in self._operations.claim_rounds(eligible):
            bucket = control_ops if claimed[0].kind in CONTROL_KINDS else data_ops
            for rec in claimed:
                bucket.append(
                    {
                        "name": name,
                        "slot": eligible[name],
                        "ordinal": rec.ordinal,
                        "request_id": rec.request_id,
                        "kind": rec.kind,
                        "payload": rec.payload,
                    }
                )
        return {"data_ops": data_ops, "control_ops": control_ops}

    def complete_operations(self, results: list[dict]) -> None:
        self._operations.complete(results)

    def operation_queue_depths(self) -> dict[str, int]:
        return self._operations.depths()
