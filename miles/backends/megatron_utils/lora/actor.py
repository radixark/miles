from contextlib import ExitStack
from pathlib import Path

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import write_sampler_export
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
            result = lora_model.run_forward_backward(self.args, batch_id, self.model, self.adapter_slots, rollout_data)
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
            result = lora_model.run_forward_backward(
                self.args, batch_id, self.model, self.adapter_slots, rollout_data, forward_only=True
            )
        self.prof.step(rollout_id=batch_id)
        return result

    @with_logs
    def load_slot(
        self,
        slot: int,
        rank: int,
        alpha: float,
        ckpt_path: str | None = None,
        load_optimizer: bool = True,
        parametrization_fields: dict | None = None,
    ) -> dict | None:
        """A source the slot cannot take is refused before the slot changes, as a user error."""
        if ckpt_path is not None:
            mismatch = lora_checkpoint.find_load_mismatch(
                self.model,
                slot,
                ckpt_path,
                self.adapter_slots.parametrization(**parametrization_fields),
                self.adapter_slots.targets,
                load_optimizer,
            )
            if mismatch is not None:
                return {"error": mismatch, "error_category": "user"}
            # load_state lands in the slot create_model claimed; it is claimed again at the source's alpha
            lora_model.unload_slot(self.adapter_slots, self.slot_optimizers.pop(slot))
        self.slot_optimizers[slot] = lora_model.load_slot(self.args, self.model, self.adapter_slots, slot, rank, alpha)
        if ckpt_path is not None:
            lora_checkpoint.load_slot(
                self.model,
                self._checkpoint_members(),
                self.slot_optimizers[slot],
                ckpt_path,
                self.adapter_slots.parametrization(**parametrization_fields),
                self.adapter_slots.targets,
                load_optimizer,
            )
        return None

    @with_logs
    def save_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """Write the slot's adapter state under ``root``, uncommitted; the gateway commits it."""
        return lora_checkpoint.save_slot(
            self._checkpoint_members(),
            slot,
            root,
            name,
            tenant_digest,
            self.adapter_slots.parametrization(**parametrization_fields),
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
        return write_sampler_export(
            Path(root),
            name,
            self.adapter_slots,
            slot,
            parametrization,
            self.adapter_slots.targets,
            tenant_digest=tenant_digest,
        )

    @with_logs
    def unload_slot(self, slot: int) -> dict | None:
        slot_optimizer = self.slot_optimizers.pop(slot)
        lora_model.unload_slot(self.adapter_slots, slot_optimizer)
        return None
