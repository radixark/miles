from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from miles.backends.training_utils.loss_hub.entropy_control import entropy_coef_to_apply, update_adaptive_entropy


def _args(**overrides):
    values = dict(
        use_adaptive_entropy=True,
        entropy_coef=0.01,
        entropy_target=0.5,
        entropy_coef_delta=0.005,
        entropy_coef_min=0.0,
        entropy_coef_max=0.02,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_static_mode_applies_the_coefficient_as_is():
    args = _args(use_adaptive_entropy=False, entropy_coef=0.3)
    loss_dict = {"entropy_loss": 0.0}

    assert entropy_coef_to_apply(args) == 0.3
    assert update_adaptive_entropy(args, loss_dict) is None
    assert args.entropy_coef == 0.3
    assert loss_dict == {"entropy_loss": 0.0}
    assert not hasattr(args, "adaptive_entropy_coef")


def test_first_step_applies_the_starting_coefficient():
    assert entropy_coef_to_apply(_args()) == 0.01


def test_coefficient_rises_while_entropy_is_at_or_below_target_and_is_clamped():
    args = _args()

    update_adaptive_entropy(args, {"entropy_loss": 0.4})
    assert args.adaptive_entropy_coef == 0.015
    assert args.adaptive_entropy_last == 0.4
    assert entropy_coef_to_apply(args) == 0.015

    update_adaptive_entropy(args, {"entropy_loss": 0.5})
    update_adaptive_entropy(args, {"entropy_loss": 0.1})
    assert args.adaptive_entropy_coef == 0.02  # clamped at max


def test_coefficient_falls_and_bonus_is_gated_off_above_target():
    args = _args(entropy_coef=0.004)

    update_adaptive_entropy(args, {"entropy_loss": 0.9})

    assert args.adaptive_entropy_coef == 0.0  # clamped at min
    assert entropy_coef_to_apply(args) == 0.0


@pytest.mark.parametrize("entropy, expected_coef", [(0.4, 0.015), (0.5, 0.015), (0.9, 0.0)])
def test_update_reports_the_coefficient_to_apply_on_the_next_step(entropy, expected_coef):
    args = _args()
    loss_dict = {"entropy_loss": entropy, "pg_loss": 1.0}

    assert update_adaptive_entropy(args, loss_dict) is None

    assert loss_dict == {"entropy_loss": entropy, "pg_loss": 1.0, "entropy_coef": expected_coef}
    assert entropy_coef_to_apply(args) == expected_coef


def test_bonus_uses_the_adapted_coefficient_when_the_gate_reopens():
    args = _args()

    update_adaptive_entropy(args, {"entropy_loss": 0.9})
    assert args.adaptive_entropy_coef == 0.005
    assert entropy_coef_to_apply(args) == 0.0

    update_adaptive_entropy(args, {"entropy_loss": 0.4})
    assert entropy_coef_to_apply(args) == 0.01


def test_updates_preserve_the_starting_coefficient_in_checkpoint_args():
    args = _args()

    update_adaptive_entropy(args, {"entropy_loss": 0.4})
    # Megatron includes the args namespace in its checkpoint state dict.
    checkpoint = pickle.loads(pickle.dumps({"args": args}))

    assert args.entropy_coef == 0.01
    assert checkpoint["args"].entropy_coef == 0.01
    assert checkpoint["args"].adaptive_entropy_coef == 0.015


def test_missing_entropy_metric_leaves_state_alone():
    args = _args()
    loss_dict = {"pg_loss": 1.0}

    update_adaptive_entropy(args, loss_dict)

    assert loss_dict == {"pg_loss": 1.0}
    assert args.entropy_coef == 0.01
    assert not hasattr(args, "adaptive_entropy_coef")
    assert not hasattr(args, "adaptive_entropy_last")
