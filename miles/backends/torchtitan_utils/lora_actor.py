"""The torchtitan multi-LoRA actor: the Tinker gateway's slot RPCs over torchtitan's multi-slot model."""

from functools import partial
from pathlib import Path

import torch

from miles.backends.torchtitan_utils.actor import TorchtitanTrainRayActor
from miles.backends.torchtitan_utils.hf_weight_iterator import TitanHfWeightIterator
from miles.backends.torchtitan_utils.lora_checkpoint import TorchtitanAdapterSlots, TorchtitanSlotOptimizerMembers
from miles.backends.torchtitan_utils.slot_optimizer import TorchtitanSlotOptimizer, step_slot_optimizers
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import agree_on_adapter_source, load_adapter_source
from miles.backends.training_utils.data.rollout import get_batch, get_data_iterator, get_rollout_data
from miles.backends.training_utils.loss.hub.response_layout import response_layout_type
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.metrics.log_utils import aggregate_train_losses
from miles.backends.training_utils.multi_lora_actor import MultiLoRASlotRpcs
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets
from miles.utils.object_store import StoreObjectRef

# the keys a Tinker loss reads, beside what get_batch adds
_SLOT_PASS_KEYS = [
    "tokens",
    "total_lengths",
    "response_lengths",
    "loss_masks",
    "log_probs",
    "advantages",
    "rollout_log_probs",
    "max_seq_lens",
    "loss_weights",
    "target_tokens",
    "sample_indices",
]


class TorchtitanMultiLoRATrainRayActor(MultiLoRASlotRpcs, TorchtitanTrainRayActor):
    # the gateway serves no rollouts, and each slot's optimizer steps per request at the request's lr: the engine's
    # own schedule never steps
    def _steps_per_rollout(self) -> int:
        return 1

    def _lr_total_steps(self) -> int:
        return 1

    def _init_training_state(self) -> None:
        # adapters reach the engines as sampler exports the gateway publishes, not through a weight update
        self.weight_updater = None
        self.slot_optimizers: dict[int, TorchtitanSlotOptimizer] = {}
        iterator = TitanHfWeightIterator.build(
            self.args,
            self.trainer,
            required_placement=WeightUpdatePlacement(gather_pp=True),
            model_name=None,
            quantization_config=None,
        )
        self.adapter_slots = TorchtitanAdapterSlots(
            self.trainer.model_parts,
            iterator,
            self.trainer.engine.state_dict_adapter,
            get_hf_lora_targets(load_hf_config(self.args.hf_checkpoint).to_dict()),
            multi_lora=True,
        )

    def _slot_pass(self, batch_id: int, rollout_data_ref: StoreObjectRef, *, forward_only: bool) -> dict:
        """One work unit; gradients add to what the slots already accumulated, until their optim_step."""
        rollout_data, store_get_result = get_rollout_data(
            args=self.args,
            rollout_data_ref=rollout_data_ref,
            output_store_read=self.output_store_read,
            train_parallel_config=self.train_parallel_config,
            sample_device=response_layout_type(self.args).sample_device,
        )
        with store_get_result:
            data_iterators, num_microbatches = get_data_iterator(self.args, self.model_parts, rollout_data)
            assert len(num_microbatches) == 1, "a work unit is a single forward/backward pass"
            data_iterator = data_iterators[0]
            data_iterator.reset()
            # the trainer tallies each pass's loss here; zero_grad, which starts the tally in RL, would drop the
            # slots' accumulated gradients
            self.trainer.engine.loss = torch.zeros((), device=self.trainer.device)
            loss = partial(_slot_loss, self.args, num_microbatches[0])
            logs = []
            for _ in range(num_microbatches[0]):
                batch = get_batch(
                    data_iterator,
                    _SLOT_PASS_KEYS,
                    self.args.data_pad_size_multiplier,
                    self.args.qkv_format,
                    get_position_ids=True,
                )
                self.adapter_slots.route_tokens(batch["adapter_token_counts"])
                if forward_only:
                    with torch.no_grad():
                        logs += self.trainer.run_forward([batch], lambda logits, b: loss(logits, b)[1])
                else:
                    logs += self.trainer.run_forward_backward([batch], loss)
        self.prof.step(rollout_id=batch_id)
        per_datum = [output for log in logs for output in log["per_datum"]]
        if get_parallel_state().is_pp_last_stage:
            return {"metrics": aggregate_train_losses(logs), "per_datum": per_datum}
        return {"metrics": {}, "per_datum": per_datum}

    def _step_slots(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        return step_slot_optimizers(self.slot_optimizers, adam_params_by_slot)

    def _new_slot_optimizer(self, slot: int) -> TorchtitanSlotOptimizer:
        return TorchtitanSlotOptimizer(slot, self.adapter_slots.slot_params(slot))

    def _check_load_source(self, slot: int, path: str, parametrization, load_optimizer: bool, rank: int) -> tuple:
        kind, mismatch, checked = agree_on_adapter_source(
            Path(path),
            parametrization,
            self.adapter_slots.targets,
            load_optimizer=load_optimizer,
            run_shapes=self.adapter_slots.global_shapes(rank),
            head_rows=self.adapter_slots.head_rows,
        )
        if mismatch is None and kind.is_read_into_model():
            # no in-model loader reads a torchtitan slot
            mismatch = f"{path} is a {kind.value} source, which this backend cannot load"
        return mismatch, (None if mismatch is not None else checked)

    def _load_source(self, slot: int, path: str, parametrization, load_optimizer: bool, checked) -> None:
        load_adapter_source(
            Path(path),
            self._checkpoint_members(),
            slot,
            parametrization,
            self.adapter_slots.targets,
            load_optimizer=load_optimizer,
            in_model_loaders_by_kind={},
            checked=checked,
        )

    def _checkpoint_members(self) -> TorchtitanSlotOptimizerMembers:
        return TorchtitanSlotOptimizerMembers(self.adapter_slots, self.slot_optimizers)


def _slot_loss(args, num_microbatches: int, logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    loss, _normalizer, log_dict = loss_function(
        args=args, batch=batch, num_microbatches=num_microbatches, logits=logits, apply_megatron_loss_scaling=False
    )
    return loss, log_dict
