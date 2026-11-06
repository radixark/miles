from contextlib import ExitStack
from pathlib import Path

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    save_adapter_dir,
    save_external_export,
    save_sampler_export,
)
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.backends.training_utils.data.rollout import get_rollout_data
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs


class MultiLoRATrainRayActor(MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        self.slot_optimizers: dict[int, SlotOptimizer] = {}
        self._init_weight_updater_and_publisher(update_weights=False, publish_snapshots=True)

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(
                args=self.args,
                rollout_data_ref=rollout_data_ref,
                output_store_read=self.output_store_read,
                train_parallel_config=get_parallel_state().train_parallel_config(supports_precomputed_schedule=True),
            )
            stack.enter_context(store_get_result)
            result = lora_model.run_forward_backward(self.args, batch_id, self.model, rollout_data)
        self.prof.step(rollout_id=batch_id)
        return result

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        self._heartbeat.bump()
        return lora_model.optim_step(self.slot_optimizers, adam_params_by_slot)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        """Same loss pass as forward_backward, without the backward: the Tinker
        forward() contract returns the requested loss per datum."""
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(
                args=self.args,
                rollout_data_ref=rollout_data_ref,
                output_store_read=self.output_store_read,
                train_parallel_config=get_parallel_state().train_parallel_config(supports_precomputed_schedule=True),
            )
            stack.enter_context(store_get_result)
            result = lora_model.run_forward_backward(self.args, batch_id, self.model, rollout_data, forward_only=True)
        self.prof.step(rollout_id=batch_id)
        return result

    @with_logs
    def load_slot(self, slot: int, rank: int, alpha: float, source: dict | None = None) -> dict | None:
        """``source``: ``path``, ``load_optimizer`` and the source's ``parametrization_fields``; None starts the slot
        fresh. A source the slot cannot take is refused before the slot changes, as a user error."""
        if source is not None:
            parametrization = self.adapter_slots.parametrization(**source["parametrization_fields"])
            mismatch, checked = lora_checkpoint.check_load_source(
                self.model,
                slot,
                source["path"],
                parametrization,
                self.adapter_slots.targets,
                source["load_optimizer"],
                run_shapes=self.adapter_slots.global_shapes(rank),
                head_rows=self.adapter_slots.head_rows,
            )
            if mismatch is not None:
                return {"error": mismatch, "error_category": "user"}
        self.slot_optimizers[slot] = lora_model.load_slot(self.args, self.model, slot, rank, alpha)
        if source is not None:
            lora_checkpoint.load_slot(
                self.model,
                self._checkpoint_members(),
                self.slot_optimizers[slot],
                source["path"],
                parametrization,
                self.adapter_slots.targets,
                source["load_optimizer"],
                checked=checked,
            )
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

    def _checkpoint_members(self) -> "lora_checkpoint.LayerWiseSlotMembers":
        return lora_checkpoint.LayerWiseSlotMembers(self.adapter_slots, self.slot_optimizers)

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
        slot_optimizer = self.slot_optimizers.pop(slot)
        lora_model.unload_slot(self.model, slot_optimizer)
        return None
