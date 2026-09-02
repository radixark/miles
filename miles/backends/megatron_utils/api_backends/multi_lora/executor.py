"""Execute Multi-LoRA optimizer operations in slot-sorted collective order.
Lease bindings are validated against local residency before any mutation."""

import logging
from dataclasses import dataclass
from typing import Any

from miles.backends.megatron_utils.api_backends.multi_lora.optimizer import step_adapter_slots, zero_adapter_slot_grads
from miles.backends.training_utils.operation_execution import StepRequest
from miles.ray.multi_lora.residency import ResidentBinding
from miles.utils.operation_contract import BatchExecutionLease

logger = logging.getLogger(__name__)


@dataclass
class MultiLoraParameterExecutor:
    model: Any
    optimizer: Any
    loaded_adapters: dict

    def discard_many(self, lease: BatchExecutionLease[ResidentBinding], operation_ids: list[str]) -> dict[str, dict]:
        outcomes: dict[str, dict] = {}
        targets: list[tuple[int, str]] = []
        for operation_id in operation_ids:
            slot, refusal = self._resolve_slot(lease, operation_id)
            if refusal is not None:
                outcomes[operation_id] = refusal
                continue
            targets.append((slot, operation_id))
        for slot, operation_id in sorted(targets):
            zero_adapter_slot_grads(self.model, slot)
            outcomes[operation_id] = dict(ok=True, gradient_window_consumed=True)
        return outcomes

    def step_many(self, lease: BatchExecutionLease[ResidentBinding], requests: list[StepRequest]) -> dict[str, dict]:
        outcomes: dict[str, dict] = {}
        adam_by_slot: dict[int, dict] = {}
        operation_by_slot: dict[int, str] = {}
        duplicate_slots: set[int] = set()
        for request in requests:
            slot, refusal = self._resolve_slot(lease, request.operation_id)
            if refusal is not None:
                outcomes[request.operation_id] = refusal
                continue
            if slot in operation_by_slot:
                duplicate_slots.add(slot)
                continue
            adam_by_slot[slot] = request.adam_params
            operation_by_slot[slot] = request.operation_id
        if duplicate_slots:
            for slot in duplicate_slots:
                adam_by_slot.pop(slot, None)
                operation_by_slot.pop(slot, None)
            for request in requests:
                binding = lease.binding_of(request.operation_id)
                if binding is not None and binding.training_slot in duplicate_slots:
                    outcomes[request.operation_id] = dict(
                        ok=False,
                        error=(
                            f"operation '{request.operation_id}' shares physical slot "
                            f"{binding.training_slot} with another operation in this batch; "
                            "refusing every operation on that slot"
                        ),
                        category="server",
                    )
        if adam_by_slot:
            grad_norms, vetoed, norm_blind = step_adapter_slots(self.optimizer, self.model, adam_by_slot)
            for slot, operation_id in operation_by_slot.items():
                if slot in vetoed:
                    outcomes[operation_id] = dict(
                        ok=False,
                        error="non-finite gradients; step vetoed and gradients cleared",
                        category="server",
                        gradient_window_consumed=True,
                    )
                elif slot in norm_blind:
                    outcomes[operation_id] = dict(
                        ok=False,
                        error="grads exist but no grad-norm sources (param-flagging bug); step refused, grads cleared",
                        category="server",
                        gradient_window_consumed=True,
                    )
                else:
                    outcomes[operation_id] = dict(
                        ok=True,
                        gradient_window_consumed=True,
                        result=dict(
                            grad_norm=grad_norms.get(slot),
                            learning_rate=adam_by_slot[slot].get("learning_rate", 1e-4),
                        ),
                    )
        return outcomes

    def _resolve_slot(self, lease, operation_id: str) -> tuple[int | None, dict | None]:
        """Lease -> local residency validation; (slot, None) or (None, outcome)."""
        binding = lease.binding_of(operation_id)
        if binding is None:
            return None, dict(
                ok=False, error=f"operation '{operation_id}' has no binding in the batch lease", category="server"
            )
        name, registration_id = binding.registration_key
        run = self.loaded_adapters.get(name)
        if run is None or run.registration_id != registration_id or run.slot != binding.training_slot:
            return None, dict(
                ok=False,
                error=f"adapter '{name}' is not resident in slot {binding.training_slot}",
                category="server",
            )
        return binding.training_slot, None
