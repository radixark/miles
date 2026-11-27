"""The Tinker gateway's slot RPCs, shared by every backend's multi-LoRA train actor."""

from pathlib import Path
from typing import Any

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    save_adapter_dir,
    save_external_export,
    save_sampler_export,
)
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs


class MultiLoRASlotRpcs:
    """The slot RPCs over a backend's ``adapter_slots`` (``AdapterSlots``), its per-slot optimizers and the hooks
    below. A slot's optimizer lives from ``load_slot`` to ``unload_slot``."""

    adapter_slots: Any
    slot_optimizers: dict[int, Any]

    def _slot_pass(self, batch_id: int, rollout_data_ref: StoreObjectRef, *, forward_only: bool) -> dict:
        """One work unit's loss pass: ``{"metrics", "per_datum"}``; gradients add up until the slots step."""
        raise NotImplementedError

    def _step_slots(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        raise NotImplementedError

    def _new_slot_optimizer(self, slot: int):
        """The optimizer of a slot just claimed; it has ``zero_grads``."""
        raise NotImplementedError

    def _check_load_source(self, slot: int, path: str, parametrization, load_optimizer: bool, rank: int) -> tuple:
        """Collective: ``(why the source cannot load into the slot or None, the checked source)``."""
        raise NotImplementedError

    def _load_source(self, slot: int, path: str, parametrization, load_optimizer: bool, checked) -> None:
        raise NotImplementedError

    def _checkpoint_members(self):
        raise NotImplementedError

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        self._heartbeat.bump()
        return self._slot_pass(batch_id, rollout_data_ref, forward_only=False)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        """Same loss pass as forward_backward, without the backward: the Tinker
        forward() contract returns the requested loss per datum."""
        self._heartbeat.bump()
        return self._slot_pass(batch_id, rollout_data_ref, forward_only=True)

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        self._heartbeat.bump()
        return self._step_slots(adam_params_by_slot)

    @with_logs
    def load_slot(self, slot: int, rank: int, alpha: float, source: dict | None = None) -> dict | None:
        """``source``: ``path``, ``load_optimizer`` and the source's ``parametrization_fields``; None starts the slot
        fresh. A source the slot cannot take is refused before the slot changes, as a user error."""
        if source is not None:
            parametrization = self.adapter_slots.parametrization(**source["parametrization_fields"])
            mismatch, checked = self._check_load_source(
                slot, source["path"], parametrization, source["load_optimizer"], rank
            )
            if mismatch is not None:
                return {"error": mismatch, "error_category": "user"}
            # load_state lands in the slot create_model claimed; it is claimed again at the source's alpha
            self._unload(slot)
        # the gateway takes no per-adapter seed
        self.adapter_slots.init(slot, rank, alpha, seed=None)
        self.slot_optimizers[slot] = self._new_slot_optimizer(slot)
        self.slot_optimizers[slot].zero_grads()
        if source is not None:
            self._load_source(slot, source["path"], parametrization, source["load_optimizer"], checked)
        return None

    @with_logs
    def save_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """Write the slot's adapter state under ``root``, uncommitted; the gateway commits it."""
        return save_adapter_dir(
            Path(root),
            name,
            self._checkpoint_members(),
            slot,
            self.adapter_slots.parametrization(**parametrization_fields),
            tenant_digest=tenant_digest,
        )

    @with_logs
    def export_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """Write the slot's sampler export under ``root``, uncommitted; the gateway commits it."""
        self._heartbeat.bump()
        parametrization = self.adapter_slots.parametrization(**parametrization_fields)
        assert parametrization.lora_rank == self.adapter_slots.lora_rank(
            slot
        ), "the gateway and the slot disagree on rank"
        return save_sampler_export(
            root=Path(root),
            name=name,
            members=self.adapter_slots,
            slot=slot,
            parametrization=parametrization,
            targets=self.adapter_slots.targets,
            tenant_digest=tenant_digest,
            lora_dropout=self.args.lora_dropout,
        )

    @with_logs
    def export_external_slot(
        self,
        slot: int,
        root: str,
        name: str,
        tenant_digest: str,
        parametrization_fields: dict,
        age_encryption_recipients: list[str],
    ) -> UncommittedAdapterDir:
        """Write the slot's official Tinker export under ``root``, uncommitted; the gateway commits it."""
        self._heartbeat.bump()
        parametrization = self.adapter_slots.parametrization(**parametrization_fields)
        assert parametrization.lora_rank == self.adapter_slots.lora_rank(
            slot
        ), "the gateway and the slot disagree on rank"
        return save_external_export(
            Path(root),
            name,
            self._checkpoint_members(),
            slot,
            parametrization,
            self.adapter_slots.targets,
            tenant_digest=tenant_digest,
            age_encryption_recipients=age_encryption_recipients,
            lora_dropout=self.args.lora_dropout,
        )

    @with_logs
    def unload_slot(self, slot: int) -> dict | None:
        self._unload(slot)
        return None

    def _unload(self, slot: int) -> None:
        """The slot's optimizer state dies with the instance the caller drops."""
        slot_optimizer = self.slot_optimizers.pop(slot)
        self.adapter_slots.clear(slot)
        slot_optimizer.zero_grads()
