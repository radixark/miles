from contextlib import ExitStack

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
from miles.backends.training_utils.data import get_rollout_data
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.snapshot_publisher import WeightPublisher
from miles.utils.lora import build_lora_config
from miles.utils.multi_lora import AdapterSpec
from miles.utils.ray_utils import Box
from miles.utils.tracking_utils.structured_log import with_logs


class MultiLoRATrainRayActor(MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        args = self.args
        self.slot_optimizers: dict[int, SlotOptimizer] = {}
        iterator = get_hf_weight_iterator(
            args,
            self.model,
            required_placement=WeightUpdatePlacement(gather_pp=True),
            model_name=type(self.hf_config).__name__.lower() if args.model_name is None else args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )
        self.weight_publisher = WeightPublisher(iterator, build_lora_config(args, target_modules=args.hf_lora_targets))

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: Box) -> dict:
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return lora_model.run_forward_backward(self.args, batch_id, self.model, rollout_data)

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        self._heartbeat.bump()
        return lora_model.optim_step(self.slot_optimizers, adam_params_by_slot)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: Box) -> dict:
        """Same loss pass as forward_backward, without the backward: the Tinker
        forward() contract returns the requested loss per datum."""
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return lora_model.run_forward_backward(self.args, batch_id, self.model, rollout_data, forward_only=True)

    @with_logs
    def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        self.slot_optimizers[slot] = lora_model.load_slot(self.args, self.model, slot, rank, alpha)
        if ckpt_path is not None:
            lora_checkpoint.load_slot(self.model, self.slot_optimizers[slot], ckpt_path, load_optimizer)

    @with_logs
    def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> None:
        lora_checkpoint.save_slot(self.model, self.slot_optimizers[slot], path, metadata=metadata)

    @with_logs
    def export_slot(self, slot: int, rank: int, alpha: float, path: str, metadata: dict | None = None) -> None:
        """Write the slot's adapter as an engine-loadable dir."""
        self._heartbeat.bump()
        self.weight_publisher.publish_adapter(AdapterSpec(slot=slot, rank=rank, alpha=alpha), path, metadata=metadata)

    @with_logs
    def unload_slot(self, slot: int) -> dict | None:
        slot_optimizer = self.slot_optimizers.pop(slot)
        lora_model.unload_slot(self.model, slot_optimizer)
        return None
