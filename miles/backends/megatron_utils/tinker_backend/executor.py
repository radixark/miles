"""Multi-LoRA concrete of the generic ParameterExecutor port
(codex-rollout-fullparameter-design-0810 §3.5): a thin adapter over the
existing slot primitives — selective grad discard, per-slot Adam step with
the all-rank veto, slot-sorted collective order.

Bindings resolve EXCLUSIVELY from the batch execution lease, and each one is
validated against this rank's locally loaded adapters (exact name,
registration id, and slot) before any weights/optimizer/grad mutation; a
stale binding yields a server-error outcome for that operation, never a
mutation of another tenant's state. Outcomes key by operation ID only."""

import logging
from dataclasses import dataclass
from typing import Any

from miles.backends.megatron_utils.tinker_backend.optimizer import step_adapter_slots, zero_adapter_slot_grads
from miles.backends.training_utils.tinker_execution import StepRequest
from miles.ray.tinker_backend.residency import ResidentBinding
from miles.utils.tinker_backend import BatchExecutionLease

logger = logging.getLogger(__name__)


@dataclass
class MultiLoraParameterExecutor:
    model: Any
    optimizer: Any
    loaded_adapters: dict

    def discard_many(self, lease: BatchExecutionLease[ResidentBinding], operation_ids: list[str]) -> dict[str, dict]:
        """Discard the listed operations' gradient windows (poisoned steps):
        zero each slot's partial gradient sum on this rank, in slot-sorted
        order so every rank's sequence matches."""
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
        """Apply each operation's AdamParams and step its slot's accumulated
        gradient sum (step_adapter_slots owns the slot-sorted collective order
        and the unanimous non-finite veto). Every outcome carries
        ``gradient_window_consumed``: True for a step or a veto (both leave
        the slot's gradients cleared on every rank), absent for a refusal
        that never touched them."""
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
                # Two operations bound to one physical slot in one batch: the
                # generic lease contract has no answer for which AdamParams
                # win, and rekeying by slot would silently drop one. Refuse
                # every operation on that slot deterministically (same
                # decision on every rank), with no gradient mutation.
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
                        error=(
                            "gradient-norm collection is structurally empty while gradients exist "
                            "(parameter-flagging bug in the parameterization); step refused and "
                            "gradients cleared"
                        ),
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
