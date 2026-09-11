"""Accumulate gradients across commands and step only the requested LoRA slots."""

from argparse import Namespace
from collections.abc import Sequence

from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora.optimizer import (
    SlotOptimizer,
    reset_grad_metadata_keep_grads,
    step_slot_optimizers,
)
from miles.backends.megatron_utils.model import run_forward_backward_pass, setup_train_iteration_config
from miles.backends.training_utils.data import get_data_iterator
from miles.backends.training_utils.log_utils import aggregate_train_losses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase
from miles.utils.types import RolloutBatch


def run_loss_pass(
    args: Namespace,
    batch_id: int,
    model: Sequence[DDP],
    rollout_data: RolloutBatch,
    *,
    forward_only: bool = False,
) -> dict:
    data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)
    assert len(num_microbatches) == 1, "a work unit is a single forward/backward pass"

    for iterator in data_iterator:
        iterator.reset()
    for model_chunk in model:
        model_chunk.train()
    # disable_optimizer: bf16 loss scaling is the identity, so the pass needs no optimizer
    setup_train_iteration_config(args, model, None, disable_optimizer=True)
    if not forward_only:
        reset_grad_metadata_keep_grads(model)

    dumper_phase_util = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=batch_id)
    losses_reduced = run_forward_backward_pass(
        args,
        dumper_phase_util,
        data_iterator,
        model,
        num_microbatches[0],
        num_rollouts=None,
        forward_only=forward_only,
    )
    per_datum_outputs = [output for microbatch in losses_reduced for output in microbatch["per_datum"]]
    dumper_phase_util.finalize(model)

    if get_parallel_state().is_pp_last_stage:
        return {"metrics": aggregate_train_losses(losses_reduced, None), "per_datum": per_datum_outputs}
    return {"metrics": {}, "per_datum": per_datum_outputs}


def optim_step(
    slot_optimizers: dict[int, SlotOptimizer],
    adam_params_by_slot: dict[int, dict],
) -> dict[int, dict]:
    stepped = {slot: slot_optimizers[slot] for slot in adam_params_by_slot}
    return step_slot_optimizers(stepped, adam_params_by_slot)


def zero_grads(slot_optimizer: SlotOptimizer) -> None:
    slot_optimizer.zero_grads()


def load_slot(args: Namespace, model: Sequence[DDP], slot: int, rank: int, alpha: float) -> SlotOptimizer:
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot

    init_adapter_slot(model, slot, rank=rank, alpha=alpha)
    slot_optimizer = SlotOptimizer(args, model, slot)
    slot_optimizer.zero_grads()
    return slot_optimizer


def unload_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer) -> None:
    """The caller drops the instance afterwards; its state dies with it."""
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    clear_adapter_slot(model, slot_optimizer.slot)
    slot_optimizer.zero_grads()
