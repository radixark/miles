"""Per-adapter LR/WD schedules for multi-LoRA: one ``OptimizerParamScheduler`` per adapter slot, positioned by
the adapter's own trained samples. Adapters without a known ``num_step`` warm up, then hold ``--lr`` constant."""

import logging
from argparse import Namespace

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

logger = logging.getLogger(__name__)


class _SlotParamGroups:
    """Minimal optimizer facade: the scheduler only reads ``.param_groups``."""

    def __init__(self, param_groups: list[dict]):
        self.param_groups = param_groups


def build_slot_scheduler(args: Namespace, optimizer, adapter, resume_step: int) -> OptimizerParamScheduler:
    """Build the slot's scheduler and position it at the adapter's committed
    samples. Rebuilt on every adapter load, so slot reuse starts fresh."""
    from miles.backends.megatron_utils.multi_lora_optimizer import _slot_children

    groups = [group for child in _slot_children(optimizer, adapter.slot) for group in child.param_groups]
    samples_per_step = adapter.config.adapter_global_batch_size
    num_step = adapter.config.num_step

    decay_steps = num_step * samples_per_step if num_step is not None else None
    if args.lr_warmup_fraction is not None and decay_steps is not None:
        lr_warmup_steps = args.lr_warmup_fraction * decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * samples_per_step
    if decay_steps is None:
        # No horizon: warm up, then hold constant. The decay steps only need
        # to satisfy the scheduler's warmup < decay invariant.
        lr_decay_style = "constant"
        decay_steps = int(lr_warmup_steps) + 1
    else:
        lr_decay_style = args.lr_decay_style

    scheduler = OptimizerParamScheduler(
        _SlotParamGroups(groups),
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=decay_steps,
        lr_decay_style=lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=decay_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        wsd_decay_steps=(
            args.lr_wsd_decay_iters * samples_per_step
            if lr_decay_style == "WSD" and args.lr_wsd_decay_iters is not None
            else None
        ),
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )
    if resume_step:
        scheduler.step(increment=resume_step * samples_per_step)
    return scheduler


def install_slot_scheduler(args: Namespace, optimizer, adapter, resume_step: int) -> None:
    """Attach the adapter's scheduler to the optimizer, keyed by slot."""
    if not hasattr(optimizer, "miles_slot_schedulers"):
        optimizer.miles_slot_schedulers = {}
    optimizer.miles_slot_schedulers[adapter.slot] = build_slot_scheduler(args, optimizer, adapter, resume_step)


def drop_slot_scheduler(optimizer, slot: int) -> None:
    """Detach a retired slot's scheduler (the next tenant installs its own)."""
    getattr(optimizer, "miles_slot_schedulers", {}).pop(slot, None)


def step_slot_schedulers(optimizer, step_batch_sizes: dict[int, int]) -> dict[int, float]:
    """Advance exactly the stepped slots' schedules by their batch samples.
    Returns slot -> new learning rate, for logging."""
    lr_by_slot: dict[int, float] = {}
    for slot, batch_size in step_batch_sizes.items():
        scheduler = optimizer.miles_slot_schedulers[slot]
        scheduler.step(increment=batch_size)
        if scheduler.optimizer.param_groups:  # empty on ranks owning none of the slot's params
            lr_by_slot[slot] = scheduler.optimizer.param_groups[0]["lr"]
    return lr_by_slot
