"""Command-grained training primitives over the multi-LoRA slot machinery.

forward_backward accumulates slot gradients across calls; optim_step consumes
them for the requested slots only. Hyperparameters arrive per optim_step call;
there is no scheduler.
"""

import logging
from argparse import Namespace
from collections.abc import Sequence

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.lora.optimizer import (
    _slot_children,
    reset_grad_metadata_keep_grads,
    step_adapter_slots,
    zero_adapter_slot_grads,
)
from miles.backends.megatron_utils.lora.slots import zero_optimizer_state_for_adapter
from miles.backends.megatron_utils.model import run_forward_backward_pass, setup_train_iteration_config
from miles.backends.training_utils.data import get_data_iterator
from miles.backends.training_utils.log_utils import aggregate_train_losses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def forward_backward(
    args: Namespace,
    unit_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: RolloutBatch,
) -> dict:
    data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)
    assert len(num_microbatches) == 1, "a work unit is a single forward/backward pass"

    for iterator in data_iterator:
        iterator.reset()
    for model_chunk in model:
        model_chunk.train()
    setup_train_iteration_config(args, model, optimizer, disable_optimizer=False)
    reset_grad_metadata_keep_grads(model)

    dumper_phase_util = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=unit_id)
    losses_reduced = run_forward_backward_pass(
        args, dumper_phase_util, data_iterator, model, num_microbatches[0], num_rollouts=None
    )
    dumper_phase_util.finalize(model)

    if get_parallel_state().is_pp_last_stage:
        return aggregate_train_losses(losses_reduced, None)
    return {}


def optim_step(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    adam_params_by_slot: dict[int, dict],
) -> dict[int, float]:
    for slot, adam_params in adam_params_by_slot.items():
        _apply_adam_params(optimizer, slot, adam_params)
    # batch size 1: grads step as accumulated; normalization is the client's loss weights
    return step_adapter_slots(
        optimizer,
        model,
        {slot: 1 for slot in adam_params_by_slot},
        clip_grad=args.clip_grad,
    )


def _apply_adam_params(optimizer: MegatronOptimizer, slot: int, adam_params: dict) -> None:
    # AdamParams is materialized at the boundary; a missing key is an encoder bug
    for child in _slot_children(optimizer, slot):
        for group in child.param_groups:
            group["lr"] = adam_params["learning_rate"]
            group["betas"] = (adam_params["beta1"], adam_params["beta2"])
            group["eps"] = adam_params["eps"]
            group["weight_decay"] = adam_params["weight_decay"]


def load_slot(model: Sequence[DDP], optimizer: MegatronOptimizer, slot: int, rank: int, alpha: float) -> None:
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot

    init_adapter_slot(model, slot, rank=rank, alpha=alpha)
    zero_adapter_slot_grads(model, slot)
    zero_optimizer_state_for_adapter(optimizer, model, slot)
    optimizer.reload_model_params()


def unload_slot(model: Sequence[DDP], optimizer: MegatronOptimizer, slot: int) -> None:
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    clear_adapter_slot(model, slot)
    zero_adapter_slot_grads(model, slot)
    zero_optimizer_state_for_adapter(optimizer, model, slot)
    optimizer.reload_model_params()
