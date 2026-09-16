from contextlib import ExitStack

import ray

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.data import get_data_iterator, get_rollout_data
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.ray_utils import Box
from miles.utils.tracking_utils.structured_log import with_logs


class MultiLoRATrainRayActor(MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        self.slot_optimizers: dict[int, SlotOptimizer] = {}

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: Box) -> dict:
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return lora_model.run_forward_backward(self.args, batch_id, self.model, rollout_data)

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, float]:
        self._heartbeat.bump()
        return lora_model.optim_step(self.args, self.slot_optimizers, adam_params_by_slot)

    @with_logs
    def forward_only_logprobs(self, batch_id: int, rollout_data_ref: Box) -> Box | None:
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
            outputs = self.compute_log_prob(data_iterator, num_microbatches, rollout_id=batch_id)
        if not get_parallel_state().is_pp_last_stage:
            return None
        return Box(ray.put({key: [t.cpu() for t in tensors] for key, tensors in outputs.items()}))

    @with_logs
    def load_slot(self, slot: int, rank: int, alpha: float) -> None:
        self.slot_optimizers[slot] = lora_model.load_slot(self.args, self.model, slot, rank, alpha)

    @with_logs
    def unload_slot(self, slot: int) -> None:
        lora_model.unload_slot(self.model, self.slot_optimizers.pop(slot))
