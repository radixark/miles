from contextlib import ExitStack

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.data.rollout import get_rollout_data
from miles.backends.training_utils.multi_lora_actor import MultiLoRASlotRpcs
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.object_store import StoreObjectRef


class MultiLoRATrainRayActor(MultiLoRASlotRpcs, MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        self.slot_optimizers: dict[int, SlotOptimizer] = {}
        self._init_weight_updater_and_publisher(update_weights=False, publish_snapshots=True)

    def _slot_pass(self, batch_id: int, rollout_data_ref: StoreObjectRef, *, forward_only: bool) -> dict:
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(
                args=self.args,
                rollout_data_ref=rollout_data_ref,
                output_store_read=self.output_store_read,
                train_parallel_config=get_parallel_state().train_parallel_config(supports_precomputed_schedule=True),
            )
            stack.enter_context(store_get_result)
            result = lora_model.run_forward_backward(
                self.args, batch_id, self.model, self.adapter_slots, rollout_data, forward_only=forward_only
            )
        self.prof.step(rollout_id=batch_id)
        return result

    def _step_slots(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        return lora_model.optim_step(self.slot_optimizers, adam_params_by_slot)

    def _new_slot_optimizer(self, slot: int) -> SlotOptimizer:
        return SlotOptimizer(self.args, self.model, slot, self.adapter_slots.slot_parameters(slot))

    def _check_load_source(self, slot: int, path: str, parametrization, load_optimizer: bool, rank: int) -> tuple:
        return lora_checkpoint.check_load_source(
            self.model,
            slot,
            path,
            parametrization,
            self.adapter_slots.targets,
            load_optimizer,
            run_shapes=self.adapter_slots.global_shapes(rank),
            head_rows=self.adapter_slots.head_rows,
        )

    def _load_source(self, slot: int, path: str, parametrization, load_optimizer: bool, checked) -> None:
        lora_checkpoint.load_slot(
            self.model,
            self._checkpoint_members(),
            self.slot_optimizers[slot],
            path,
            parametrization,
            self.adapter_slots.targets,
            load_optimizer,
            checked=checked,
        )

    def _checkpoint_members(self) -> "lora_checkpoint.LayerWiseSlotMembers":
        return lora_checkpoint.LayerWiseSlotMembers(self.adapter_slots, self.slot_optimizers)
