"""Per-adapter LR schedules: parameters inherit the global args, position is
per adapter. Pins the two defects this replaces: a late/loaded adapter no
longer inherits the shared schedule's decayed position, and resume rebuilds
the position deterministically from the adapter's committed step count."""

from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.multi_lora_scheduler import (
    install_slot_scheduler,
    step_slot_schedulers,
)

LR = 2e-5


def make_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        lr=LR,
        min_lr=0.0,
        lr_warmup_init=0.0,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_decay_style="cosine",
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        lr_wsd_decay_iters=None,
        lr_wsd_decay_style=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_optimizer(n_slots: int = 2) -> SimpleNamespace:
    children = [SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}]) for _ in range(n_slots)]
    return SimpleNamespace(
        chained_optimizers=children,
        miles_slot_child_indices={slot: [slot] for slot in range(n_slots)},
    )


def make_adapter(slot: int, num_step: int | None, samples_per_step: int = 64) -> SimpleNamespace:
    config = SimpleNamespace(num_step=num_step, adapter_global_batch_size=samples_per_step)
    return SimpleNamespace(slot=slot, name=f"a{slot}", config=config)


def slot_lr(optimizer, slot: int) -> float:
    return optimizer.chained_optimizers[slot].param_groups[0]["lr"]


def test_decaying_adapter_walks_its_own_cosine_schedule():
    optimizer = make_optimizer()
    adapter = make_adapter(slot=0, num_step=10)
    install_slot_scheduler(make_args(), optimizer, adapter, resume_step=0)

    assert slot_lr(optimizer, 0) == pytest.approx(LR)  # fresh: top of the schedule

    step_slot_schedulers(optimizer, {0: 5 * 64})  # half the horizon
    assert slot_lr(optimizer, 0) == pytest.approx(LR / 2)

    step_slot_schedulers(optimizer, {0: 100 * 64})  # far past the horizon
    assert slot_lr(optimizer, 0) == pytest.approx(0.0)  # clamped at min_lr


def test_adapter_without_num_step_holds_constant():
    optimizer = make_optimizer()
    install_slot_scheduler(make_args(), optimizer, make_adapter(slot=0, num_step=None), resume_step=0)

    step_slot_schedulers(optimizer, {0: 12345 * 64})
    assert slot_lr(optimizer, 0) == pytest.approx(LR)  # no horizon: never decays


def test_resume_position_is_deterministic_from_committed_steps():
    stepped = make_optimizer()
    install_slot_scheduler(make_args(), stepped, make_adapter(slot=0, num_step=10), resume_step=0)
    step_slot_schedulers(stepped, {0: 5 * 64})

    resumed = make_optimizer()
    install_slot_scheduler(make_args(), resumed, make_adapter(slot=0, num_step=10), resume_step=5)

    assert slot_lr(resumed, 0) == pytest.approx(slot_lr(stepped, 0))


def test_only_stepped_slots_advance():
    optimizer = make_optimizer()
    args = make_args()
    install_slot_scheduler(args, optimizer, make_adapter(slot=0, num_step=10), resume_step=0)
    install_slot_scheduler(args, optimizer, make_adapter(slot=1, num_step=10), resume_step=0)

    lr_by_slot = step_slot_schedulers(optimizer, {0: 5 * 64})

    assert set(lr_by_slot) == {0}
    assert slot_lr(optimizer, 0) == pytest.approx(LR / 2)
    assert slot_lr(optimizer, 1) == pytest.approx(LR)  # co-tenant untouched


def test_slot_reuse_installs_a_fresh_schedule():
    optimizer = make_optimizer()
    args = make_args()
    install_slot_scheduler(args, optimizer, make_adapter(slot=0, num_step=10), resume_step=0)
    step_slot_schedulers(optimizer, {0: 5 * 64})

    install_slot_scheduler(args, optimizer, make_adapter(slot=0, num_step=20), resume_step=0)
    assert slot_lr(optimizer, 0) == pytest.approx(LR)  # next tenant starts at the top


def test_warmup_ramps_from_init_lr():
    optimizer = make_optimizer()
    args = make_args(lr_warmup_iters=2)  # 2 adapter steps of warmup
    install_slot_scheduler(args, optimizer, make_adapter(slot=0, num_step=10), resume_step=0)

    assert slot_lr(optimizer, 0) == pytest.approx(0.0)  # init_lr
    step_slot_schedulers(optimizer, {0: 64})
    assert slot_lr(optimizer, 0) == pytest.approx(LR / 2)  # mid-warmup
    step_slot_schedulers(optimizer, {0: 64})
    assert slot_lr(optimizer, 0) == pytest.approx(LR)  # warmed up
