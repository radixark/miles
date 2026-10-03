"""Single-model Tinker commands backed by native full-parameter Megatron training.

Slot zero is the trainer's full model. The command signatures match the existing
multi-LoRA actor so external command backends can use the same Ray boundary.
"""

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.checkpoint import _load_checkpoint_hf
from miles.backends.megatron_utils.full_training import checkpoint
from miles.backends.megatron_utils.full_training.optimizer import GradientAccumulator
from miles.backends.megatron_utils.hf_export import save_hf_model
from miles.backends.megatron_utils.lora.model import run_forward_backward
from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states
from miles.backends.training_utils.data.rollout import get_rollout_data
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs


class FullTrainingRayActor(MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        self._init_weight_updater_and_publisher(update_weights=False, publish_snapshots=True)
        self.accumulator = GradientAccumulator(self.optimizer)
        self.model_is_active = False
        self.model_is_pristine = True

    @with_logs
    def load_slot(
        self, slot: int, rank: None, alpha: None, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        assert slot == 0 and rank is None and alpha is None, "full training has one model and no LoRA parameters"
        self._heartbeat.bump()
        self.accumulator.clear()
        if ckpt_path is not None:
            checkpoint.load(self.args, self.model, self.optimizer, ckpt_path, load_optimizer=load_optimizer)
        elif not self.model_is_pristine:
            _load_checkpoint_hf(self.model, self.optimizer, self.args, self.args.hf_checkpoint)
            reset_optimizer_states(self.optimizer)
        self._zero_grads()
        self.model_is_active = True
        self.model_is_pristine = False

    @with_logs
    def unload_slot(self, slot: int) -> None:
        assert slot == 0
        self.accumulator.clear()
        self._zero_grads()
        self.model_is_active = False

    def _zero_grads(self) -> None:
        self.optimizer.zero_grad()
        for chunk in self.model:
            chunk.zero_grad_buffer()

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        return self._execute_batch(batch_id, rollout_data_ref, forward_only=False)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        return self._execute_batch(batch_id, rollout_data_ref, forward_only=True)

    def _execute_batch(self, batch_id: int, rollout_data_ref: StoreObjectRef, *, forward_only: bool) -> dict:
        assert self.model_is_active
        self._heartbeat.bump()
        if not forward_only:
            self._zero_grads()
        data, store_result = get_rollout_data(self.args, rollout_data_ref)
        with store_result:
            result = run_forward_backward(self.args, batch_id, self.model, data, forward_only=forward_only)
        if not forward_only:
            self.accumulator.add()
        return result

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        assert self.model_is_active and set(adam_params_by_slot) == {0}
        self._heartbeat.bump()
        result = self.accumulator.step(adam_params_by_slot[0])
        self._zero_grads()
        return {0: result}

    @with_logs
    def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> dict | None:
        assert self.model_is_active and slot == 0
        self._heartbeat.bump()
        if self.accumulator.num_batches:
            return {"error": "save_state requires optim_step first; pending gradients are not checkpointed"}
        checkpoint.save(self.args, self.model, self.optimizer, path, metadata=metadata)
        return None

    @with_logs
    def export_slot(self, slot: int, rank: None, alpha: None, path: str, metadata: dict | None = None) -> None:
        assert self.model_is_active and slot == 0 and rank is None and alpha is None
        self._heartbeat.bump()
        save_hf_model(
            self.args,
            0,
            self.model,
            publisher=self.snapshot_publisher,
            path=path,
            raise_on_error=True,
            metadata=metadata,
        )
